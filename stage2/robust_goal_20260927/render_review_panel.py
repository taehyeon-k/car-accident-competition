"""Contact sheets for diagnostic review; no labels are created or changed."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parent

def main():
    panel=json.loads((OUT/'label_review_panel.json').read_text())['rows']
    rows={}
    for path in (OUT/'group_folds').glob('*_val.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line);rows[row['sample_id']]=row
    dest=OUT/'review_images';dest.mkdir(exist_ok=True)
    index=['# ENTRY diagnostic contact sheets','',
        'These sheets show existing annotations and model predictions, so this is an unblinded failure audit. No image judgment here constitutes corrected ground truth. A separate blinded review would be needed before changing training labels. Frame numbers use the cached zero-based frame filenames.','']
    failed=[]
    for item in panel:
        sid=item['sample_id'];r=rows[sid];fps=r['native_fps'];entry=r['entry_frame']
        anchors=[('Before annotated ENTRY',round(entry-.5*fps)),('Annotated ENTRY',entry),('After annotated ENTRY',round(entry+.5*fps)),
                 ('Native prediction',round(entry+item['entry_error_seconds_k1']*fps)),
                 ('Third-rate prediction',round(entry+item['entry_error_seconds_k3']*fps)),('Annotated COLLISION',r['collision_frame'])]
        sheet=Image.new('RGB',(960,430),(20,20,20));draw=ImageDraw.Draw(sheet)
        draw.text((8,6),sid+' | '+item['source']+' | '+item['entry_pattern'],fill='white')
        for i,(label,frame) in enumerate(anchors):
            frame=max(0,min(r['num_frames']-1,frame));x=(i%3)*320;y=28+(i//3)*200
            path=Path(r['frames_dir'])/f'{frame:06d}.jpg'
            if path.exists():
                with Image.open(path) as im:
                    im=im.convert('RGB');im.thumbnail((316,176));sheet.paste(im,(x+(320-im.width)//2,y))
            else:failed.append(str(path));draw.text((x+8,y+50),'Missing cached frame',fill='red')
            draw.text((x+5,y+178),f'{label}: frame {frame}',fill='white')
        sheet.save(dest/f'{sid}.jpg',quality=88)
        index.extend([f'## {sid}',f'![{sid}](review_images/{sid}.jpg)',''])
    (OUT/'REVIEW_PANEL.md').write_text('\n'.join(index)+'\n')
    (OUT/'review_images_status.json').write_text(json.dumps({'sheets':len(panel),'missing_frames':failed},indent=2)+'\n')
    print('Sheets:',len(panel),'missing frames:',len(failed))

if __name__=='__main__':main()
