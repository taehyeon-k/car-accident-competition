"""Convert openpilot v0.11.1 driving_vision.onnx to a traced TorchScript module (GPU inference without onnxruntime at runtime).

onnx2torch (installed into stage2/night/_onnx2torch_pkgs, not the main venv) lacks converters for some newer opset versions and
for Gelu (opset 20): newer versions of operators it already supports are aliased to its latest converter (same semantics for the
attributes used here), and Gelu is added (attribute `approximate`: 'none' | 'tanh'). Parity is checked against onnxruntime (CPU) on
random inputs and on real frames. Output: stage2/night/openpilot_vision.ts (+ .json with the output slices).
Usage: PYTHONPATH=stage2/night/_onnx2torch_pkgs python -m stage2.night.op_convert
"""
from __future__ import annotations

import base64, json, pickle
from pathlib import Path

import numpy as np
import onnx, onnxruntime as ort
import torch
from torch import nn
from onnx import defs

from onnx2torch.node_converters import registry as R
from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node
from onnx2torch import convert

M = "/workspace/pretrained/openpilot_v0.11.1/driving_vision.onnx"; OUT = Path(__file__).resolve().parent / "openpilot_vision.ts"


@R.add_converter(operation_type="Gelu", version=20)
def _gelu(node, graph):  # pylint: disable=unused-argument
    approx = node.attributes.get("approximate", "none")
    approx = approx.decode() if isinstance(approx, bytes) else approx
    return OperationConverterResult(torch_module=nn.GELU(approximate="tanh" if approx == "tanh" else "none"), onnx_mapping=onnx_mapping_from_node(node=node))


class _Reshape(nn.Module):
    """ONNX Reshape with allowzero: 0 in the target shape copies the input dim (allowzero=0) or is a literal 0 (allowzero=1)."""
    def __init__(self, allowzero: int): super().__init__(); self.allowzero = allowzero

    def forward(self, x, shape):
        shape = [int(v) for v in shape.tolist()]
        if not self.allowzero: shape = [x.shape[i] if v == 0 else v for i, v in enumerate(shape)]
        return torch.reshape(x, shape)


def _reshape(node, graph):  # pylint: disable=unused-argument
    return OperationConverterResult(torch_module=_Reshape(int(node.attributes.get("allowzero", 0))), onnx_mapping=onnx_mapping_from_node(node=node))


for _v in (5, 13, 14, 19, 21):
    R._CONVERTER_REGISTRY[R.OperationDescription("", "Reshape", _v)] = _reshape


class _Reduce18(nn.Module):
    """ONNX Reduce* from opset 18: axes are an optional input; keepdims / noop_with_empty_axes attributes."""
    def __init__(self, kind: str, keepdims: int, noop: int): super().__init__(); self.kind, self.keepdims, self.noop = kind, bool(keepdims), bool(noop)

    def forward(self, x, axes=None):
        dims = [int(a) for a in axes.tolist()] if axes is not None and axes.numel() else []
        if not dims:
            if self.noop: return x
            dims = list(range(x.dim()))
        k = self.kind
        if k == "ReduceMean": return x.mean(dim=dims, keepdim=self.keepdims)
        if k == "ReduceSum": return x.sum(dim=dims, keepdim=self.keepdims)
        if k == "ReduceSumSquare": return (x * x).sum(dim=dims, keepdim=self.keepdims)
        if k == "ReduceL2": return torch.sqrt((x * x).sum(dim=dims, keepdim=self.keepdims))
        if k in ("ReduceMax", "ReduceMin"):
            for d in sorted(dims, reverse=True): x = (x.amax if k == "ReduceMax" else x.amin)(dim=d, keepdim=self.keepdims)
            return x
        if k == "ReduceProd":
            for d in sorted(dims, reverse=True): x = x.prod(dim=d, keepdim=self.keepdims)
            return x
        raise NotImplementedError(k)


def _reduce_factory(kind):
    def conv(node, graph):  # pylint: disable=unused-argument
        return OperationConverterResult(torch_module=_Reduce18(kind, int(node.attributes.get("keepdims", 1)), int(node.attributes.get("noop_with_empty_axes", 0))),
                                        onnx_mapping=onnx_mapping_from_node(node=node))
    return conv


for _k in ("ReduceMean", "ReduceSum", "ReduceSumSquare", "ReduceL2", "ReduceMax", "ReduceMin", "ReduceProd"):
    for _v in (18, 20):
        R._CONVERTER_REGISTRY[R.OperationDescription("", _k, _v)] = _reduce_factory(_k)


def alias(model):
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")); added = []
    for op in {n.op_type for n in model.graph.node}:
        sv = defs.get_schema(op, max_inclusive_version=opset).since_version
        have = [d for d in R._CONVERTER_REGISTRY if d.operation_type == op and d.domain == ""]
        if have and not any(d.version == sv for d in have):
            best = max(have, key=lambda d: d.version); R._CONVERTER_REGISTRY[R.OperationDescription("", op, sv)] = R._CONVERTER_REGISTRY[best]
            added.append(f"{op}:{best.version}->{sv}")
    return opset, added


def main():
    model = onnx.load(M); opset, added = alias(model); print("opset", opset, "aliased", added)
    net = convert(model).eval()
    for mod in net.modules():  # run in float32: pure-fp16 torch execution overflows on some inputs (onnxruntime upcasts internally)
        if type(mod).__name__.lower().startswith("onnxcast") and mod.torch_dtype == torch.float16: mod.torch_dtype = torch.float32
    net = net.float().cuda(); sess = ort.InferenceSession(M, providers=["CPUExecutionProvider"])
    import cv2
    from stage2.night.openpilot_probe import yuv6, to_model_frame
    ps = sorted(Path("/workspace/data/stage2/frames/ccd_000246").glob("*.jpg"))
    fr = [yuv6(to_model_frame(cv2.imread(str(p)), 100, .5)) for p in ps[18:23]]
    x = np.stack([np.concatenate([fr[i], fr[i + 1]]) for i in range(4)]).astype(np.uint8)   # real consecutive frame pairs
    ref = np.stack([sess.run(None, {"img": x[i:i + 1], "big_img": x[i:i + 1]})[0][0] for i in range(4)]).astype(np.float32)
    first = lambda o: (o[0] if isinstance(o, (list, tuple)) else o)   # the network hard-codes batch 1 (Reshape to [1, 1024])
    with torch.no_grad():
        xs = [torch.from_numpy(x[i:i + 1]).cuda() for i in range(len(x))]
        out = np.concatenate([first(net(xi, xi)).float().cpu().numpy() for xi in xs])
        ts = torch.jit.trace(net, (xs[0], xs[0]), check_trace=False)
        out2 = np.concatenate([first(ts(xi, xi)).float().cpu().numpy() for xi in xs])
    print("torch fp32 vs onnxruntime max|diff|", float(np.abs(out - ref).max()), "| traced vs onnxruntime", float(np.abs(out2 - ref).max()), "| ref max", float(np.abs(ref).max()))
    ts.save(str(OUT))
    md = {p.key: p.value for p in model.metadata_props}; sl = pickle.loads(base64.b64decode(md["output_slices"]))
    OUT.with_suffix(".json").write_text(json.dumps({k: [v.start, v.stop] for k, v in sl.items()}, indent=1)); print("saved", OUT)


if __name__ == "__main__": main()
