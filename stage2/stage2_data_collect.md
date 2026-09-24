Task: Collect, metadata-filter, download, deduplicate, and prepare MM-AU + CausalCrash for Stage-2 manual labeling

You are working on my DACON car-accident competition project.

Your task is NOT simply to download MM-AU and CausalCrash.

Your goal is to build a reproducible pipeline that:

inspects my current repository and Stage-2 data format first,
collects the metadata for MM-AU and CausalCrash,
filters the datasets as aggressively as reasonably possible using metadata,
downloads ONLY videos that are plausible Stage-2 positives,
verifies integrity and removes duplicates,
leaves me with a high-quality manual-labeling queue,
sets up an easy labeling environment for the remaining labels,
produces a report explaining exactly what happened.

The main priority is reducing my manual filtering burden.

Do NOT assume everything written in this prompt is correct. Inspect the repository, the current Stage-2 implementation, dataset documentation, and actual downloaded metadata before making decisions.

0. Environment / repository

Current expected environment:

/workspace
/workspace/car-accident
/workspace/data
/workspace/pretrained
/workspace/cache
/workspace/runs

Main repository:

https://github.com/taehyeon-k/car-accident-competition.git

R2:

remote: r2
bucket: r2:car-accident-dataset

Important constraints:

Active work should happen on local /workspace SSD.
Do NOT use an R2 FUSE mount as active training/processing storage.
Do NOT download or restore irrelevant Stage-2/Stage-3 models, caches, or old datasets.
Do NOT pre-extract and permanently store all video frames.
Keep original videos plus metadata/manifests.
Extract individual frames only temporarily/on demand when required.
Check free disk space before every large download.
Avoid downloading the complete MM-AU video corpus unless there is absolutely no selective alternative.
Do not modify working Stage-2 model code unless required for this dataset pipeline.

Before doing anything substantial:

cd /workspace/car-accident
git status
git log -5 --oneline
find . -maxdepth 3 -type f | sort | head -n 300
df -h /workspace

Inspect all Stage-2 dataset/manifests/annotation utilities before implementing anything.

Search for things such as:

find . -iname '*stage2*' -o -iname '*label*' -o -iname '*manifest*' -o -iname '*annotation*'
grep -R "entry_frame\|collision_frame\|entry_side\|evasion_space" -n . --exclude-dir=.git

Determine the current canonical Stage-2 sample schema from the repository instead of inventing a conflicting one.

1. Exact Stage-2 dataset definition

A video is useful for this dataset ONLY when the following situation is visible.

Mandatory criteria
A. Crash video

It must contain an actual vehicle crash relevant to the ego vehicle.

Reject:

near misses with no collision,
generic dangerous driving,
accidents occurring only ahead of the ego vehicle,
unrelated traffic incidents.
B. Ego/dashcam viewpoint

The camera must be mounted in/on the ego vehicle or clearly represent an ego-centric forward driving camera.

Reject:

CCTV,
external observer footage,
phone recordings from a sidewalk,
compilation shots where the relevant accident is not ego-view,
third-person cameras.
C. Ego vehicle participates in the collision

The ego vehicle itself must physically participate in the crash.

Reject videos where the dashcam merely observes two other vehicles crashing.

D. Ego keeps its lane

Before the hazardous interaction, the ego vehicle must be traveling in its existing lane.

The event should NOT be primarily caused by the ego vehicle changing into somebody else's lane.

Normal small steering corrections or evasive steering inside the lane are not automatically a rejection.

The important causal structure is:

ego proceeds in its lane
        +
another vehicle intrudes/crosses/merges into ego path/lane
        ↓
collision involving ego
E. Another vehicle enters the ego path/lane

The crash must result from another vehicle laterally entering, merging into, cutting across, crossing, or otherwise intruding into the ego vehicle's occupied driving path.

Examples that can be useful:

adjacent vehicle cuts into ego lane
vehicle merges into ego lane
vehicle makes unsafe lane change into ego lane
vehicle crosses the road into ego path
vehicle turns across ego path
vehicle moves laterally into ego trajectory

Reject obvious cases such as:

ego rear-ends a vehicle traveling normally ahead
ego changes lane and hits another vehicle
ego loses control independently
ego drives into a stationary obstacle
accident between two non-ego vehicles
collision happens before useful entry motion is visible

Do not over-constrain unusual cases without inspecting the competition/repository semantics.

F. ENTRY must be visible

There must be enough pre-collision video to determine the moment at which the other vehicle begins entering the ego lane/path.

Do NOT treat an accident-window-start annotation automatically as ENTRY.

The exact Stage-2 ENTRY frame will be manually labeled.

Reject videos beginning after the relevant vehicle has already entered the ego path.

G. COLLISION must be visible

The actual impact must appear in the video.

The collision cannot occur after the clip ends or be completely obscured/missing.

Metadata collision timestamps may be used as an anchor, but manual labeling should verify the exact Stage-2 collision frame.

2. Overall filtering philosophy

I previously had to inspect thousands of videos manually only to discover that most were unusable.

Do NOT repeat that workflow.

Optimize the main queue for precision rather than maximum recall.

Create three states:

AUTO_REJECT
HIGH_CONFIDENCE_LABEL_QUEUE
BORDERLINE_REVIEW

HIGH_CONFIDENCE_LABEL_QUEUE should contain clips where metadata strongly supports the Stage-2 causal configuration.

BORDERLINE_REVIEW should contain potentially useful cases where metadata cannot establish one criterion reliably.

Do not mix borderline clips into the main manual-labeling queue.

Every rejection or acceptance must retain a machine-readable reason/evidence trail.

3. Directory layout

Prefer something approximately like:

/workspace/data/stage2_external/
├── metadata/
│   ├── mmau/
│   └── causalcrash/
├── raw_selected/
│   ├── mmau/
│   └── causalcrash/
├── manifests/
├── reports/
├── labeling/
└── tmp/

Do not create huge permanent frame directories.

If the existing repository already defines a better canonical path structure, follow the repository instead.

4. MM-AU collection — METADATA FIRST

MM-AU is large. Do NOT start by downloading hundreds of GB of video.

First inspect the current official MM-AU distribution and determine:

repository/dataset location,
available metadata files,
archive/shard organization,
whether videos are individual files or compressed shards,
whether selective downloads are possible,
how a selected metadata record maps to an actual video.

Look specifically for metadata fields corresponding to concepts such as:

type
weather
light
scenes
linear
accident occurred
abnormal_start_frame
abnormal_end_frame
accident_frame
t_ai
t_co
t_ae
texts
causes
measures
video/sample ID

Do NOT assume all fields exist exactly under those names. Verify the actual release.

Download metadata only first.

If hosted on Hugging Face, prefer programmatic inspection such as huggingface_hub repository listing rather than downloading everything.

Generate a local normalized metadata table.

5. MM-AU filtering

Use every useful piece of released metadata.

At minimum require:

actual accident
ego-view video
ego involved in collision
other road vehicle involved
valid pre-collision content
valid collision timestamp/window when available

Then use the accident category + texts + causes or equivalent semantic descriptions to detect concepts such as:

POSITIVE concepts:

cuts in
cut-in
lane change
changes lane
unsafe lane change
merges
merging
crosses into lane
crosses ego path
enters ego lane
moves into ego lane
turns across ego path
pulls into ego path
overtakes and enters ego path
lateral intrusion

Also look for descriptions indicating ego behavior such as:

ego proceeds straight
ego maintains lane
ego travels normally
ego continues in its lane
ego is struck by another vehicle
other vehicle enters/crosses ego path

Strong NEGATIVE concepts:

ego changes lane into other vehicle
ego overtakes and causes crash
ego rear-ends lead vehicle
ego loses control
ego runs into stationary object
ego hits pedestrian
motorcycle-only/pedestrian-only crash if incompatible with Stage 2
accident occurs only ahead of ego
ego not involved
camera is third-person
collision already underway at clip start

Do not implement filtering as only one brittle keyword list.

Use a combination of:

structured accident type
structured metadata
positive semantic evidence
negative semantic evidence
temporal metadata validity

Create an interpretable suitability score or rule set.

For every video store something like:

source
source_id
accident_type
ego_involved
other_vehicle_involved
ego_lane_keep_evidence
other_vehicle_intrusion_evidence
entry_visible_likelihood
collision_visible_likelihood
positive_evidence
negative_evidence
metadata_score
decision
decision_reason

Important:

t_ai or accident-start metadata is NOT automatically the Stage-2 ENTRY frame.

t_co or collision metadata can be used as a collision candidate/anchor, but do not blindly make it final ground truth.

6. Selective MM-AU video download

After metadata filtering:

determine exactly which candidate IDs are needed;
map candidates to video files/shards;
download ONLY what is necessary.

If videos are inside large archive shards and individual file download is impossible:

download the minimum set of shards
extract ONLY selected videos
verify extraction
delete temporary shard when safe

Do not retain giant archives unnecessarily.

Keep a download ledger:

source_id
remote_path
shard
download_status
local_path
bytes
checksum
error

Failed downloads must not crash the entire pipeline.

7. CausalCrash collection

Inspect the current official CausalCrash dataset/repository and its actual schema.

Do not assume the prior description is perfectly current.

CausalCrash may contain structured metadata describing:

agents
agent roles
vehicle types
initial motion
lane
intent
initiator
victim
observer
hazard type
causal chain
critical point
collision events/timestamps
source URL
camera/source type

Normalize it into the same filtering framework used for MM-AU.

8. CausalCrash high-precision filter

CausalCrash should allow a stricter filter than MM-AU.

Strong positive configuration resembles:

camera == ego/dashcam

ego agent:
    role == victim or clearly collision-involved
    intent == keeping_lane / proceeding normally

other vehicle:
    role == initiator
    type == road vehicle
    intent indicates:
        changing_left
        changing_right
        merging
        crossing
        swerving into ego path
        cut_in

hazard/event:
    collision
    cut_in
    unsafe_lane_change
    lateral intrusion
    crossing collision

Also inspect free-text causal descriptions.

Reject:

CCTV/non-ego viewpoint
ego is merely observer
ego is the initiating lane-changing vehicle
collision does not involve ego
only near miss
impact missing
accident begins before useful pre-entry context

For cases where structured labels contradict text, send them to BORDERLINE_REVIEW, not the high-confidence queue.

9. Download CausalCrash videos

Some CausalCrash entries may provide external source URLs instead of redistributing raw videos.

Use a robust downloader where permitted, e.g. yt-dlp, while respecting current source availability.

For every source record:

AVAILABLE_AND_DOWNLOADED
UNAVAILABLE
PRIVATE
REMOVED
DOWNLOAD_ERROR
NOT_SELECTED

Do not allow unavailable videos to block the rest of the process.

Do NOT download source videos that already failed metadata filtering.

Preserve original source URL and source ID for provenance.

10. Technical video validation

For every downloaded candidate, run ffprobe and record:

codec
width
height
duration
average FPS
frame count if reliably available
time base
start time

Check:

video decodes
duration > 0
collision metadata falls inside video
enough pre-collision duration exists
file is not corrupt

Do not permanently convert every video unless needed for browser labeling.

If a browser-compatible H.264 proxy is needed, keep:

original video
proxy video
mapping/provenance information

Do not silently alter frame numbering.

The final manual Stage-2 labels must refer to a clearly defined canonical/original frame index.

11. Deduplication

This is important because accident datasets frequently reuse internet videos.

Deduplicate:

within MM-AU
within CausalCrash
between MM-AU and CausalCrash
against existing Stage-2 data when accessible

Use at least:

exact file hash
source URL/source ID

and, when practical, a lightweight perceptual/video fingerprint based on a small number of sampled frames.

Do NOT extract entire videos into frames for deduplication.

Potential near-duplicates should be reported rather than silently deleted unless matching is extremely confident.

12. Final normalized manifest

Create a canonical manifest for every retained candidate.

Adapt it to the repository's real Stage-2 schema, but it should conceptually contain:

sample_id
source
source_id
source_url

video_path

fps
num_frames
duration_s
width
height

metadata_accident_type
metadata_accident_start
metadata_collision_time
metadata_collision_frame_candidate

ego_view_evidence
ego_involved_evidence
ego_lane_keep_evidence
other_vehicle_intrusion_evidence
entry_visible_evidence
collision_visible_evidence

filter_score
filter_decision
filter_reason
positive_evidence
negative_evidence

dedup_status

usable
entry_frame
collision_frame
entry_side
evasion_space
reject_reason
notes

The Stage-2 manual columns should initially be empty.

Produce at least:

high_confidence_label_queue.csv
borderline_review.csv
auto_rejected.csv
download_failures.csv
dedup_report.csv

JSONL versions are also acceptable/useful.

13. Optional lightweight automatic visual sanity check

Metadata filtering is the priority.

Do NOT immediately implement a huge computer-vision pipeline.

However, after high-confidence videos have been downloaded, if the repository already contains inexpensive detection/geometry utilities that can be reused safely, you may use them for a lightweight sanity check such as:

does the video actually contain vehicles?
is there enough pre-impact motion?
is the clip obviously CCTV?
does the metadata collision timestamp roughly correspond to a visible abrupt event?

Do NOT reject clips solely because an existing CV detector fails.

Use such checks as supporting evidence only.

14. Manual labeling setup

After filtering, I want to label ONLY the remaining high-quality videos.

First inspect whether the repository already contains my Stage-2 Label Studio configuration, import/export utilities, or previous annotation scripts.

REUSE existing working tooling if available.

Do not create a totally separate annotation format if the repository already has one.

The desired Stage-2 labels are:

USABLE / UNUSABLE

ENTRY frame
COLLISION frame

entry_side:
    LEFT
    RIGHT

evasion_space:
    0
    1

Exact meaning of entry_side and evasion_space must be taken from the repository/current Stage-2 documentation. Do not guess if code already defines them.

For unusable clips, allow a rejection reason such as:

ego_not_involved
ego_changed_lane
other_vehicle_not_intruding
entry_not_visible
collision_not_visible
not_dashcam
near_miss
wrong_vehicle_type
duplicate
corrupt_video
other
15. Labeling UX

I want the labeling process to be fast.

If my previous Label Studio workflow exists and works, make it easy to launch.

Prefer keyboard shortcuts.

At minimum support convenient actions for:

ENTRY
COLLISION
LEFT
RIGHT
evasion 0
evasion 1
USABLE
UNUSABLE
next video
previous video
save

For ENTRY/COLLISION labeling, exact frame selection matters.

The UI should make it easy to:

play video normally
jump near metadata collision anchor
pause
step backward/forward frame-by-frame
see the current frame index
mark ENTRY
mark COLLISION

If Label Studio cannot cleanly provide exact frame indexing for the current setup, inspect the existing code first.

Only if necessary, implement a small local labeling helper around the existing annotation format.

A good fallback would be a minimal Python web app that:

shows video playback
lets me seek
shows exact decoded frame number
supports ±1 / ±5 / ±10 frame stepping
records ENTRY/COLLISION
records LEFT/RIGHT
records evasion_space
records usable/reject reason
autosaves after each video
resumes from previous progress

Do not build an unnecessarily complicated frontend.

16. One-command launch

At the end I want a simple command such as:

bash scripts/start_stage2_external_labeling.sh

or whatever matches the repository layout.

Print the exact command and browser URL/port if a web interface is used.

The labeling state must survive restarting the server/process.

17. R2 backup

Do not upload raw unnecessary archives.

After the pipeline is verified, back up only valuable persistent artifacts.

Preferred conceptual destination:

r2:car-accident-dataset/stage2/external/

For example:

stage2/external/metadata/
stage2/external/manifests/
stage2/external/mmau/videos/
stage2/external/causalcrash/videos/
stage2/external/annotations/
stage2/external/reports/

Before uploading, check the existing R2 hierarchy and avoid conflicting with existing data.

Use checksums/size verification after upload.

Do not delete the local validated copies until remote verification succeeds.

18. Reproducibility

The entire collection/filtering process should be reproducible.

Create scripts/configuration rather than relying on ad-hoc shell history.

Prefer components conceptually like:

collect_mmau_metadata.py
collect_causalcrash_metadata.py
normalize_external_metadata.py
filter_stage2_candidates.py
download_selected_videos.py
validate_videos.py
deduplicate_candidates.py
build_label_queue.py
start_stage2_external_labeling.sh

Exact names/organization should follow the existing repository style.

Keep filtering thresholds/keywords/rules in a config file rather than burying everything inside code.

19. Do not blindly trust this prompt

This is important.

Before implementing any dataset-specific assumption:

inspect the current repository;
inspect the actual MM-AU release;
inspect the actual CausalCrash release;
inspect their current metadata schemas;
determine whether the required field actually exists.

If this prompt says a field exists but the downloaded dataset does not contain it, adapt to reality.

Do NOT fabricate metadata.

Do NOT infer a field exists merely because a paper mentioned it.

When metadata is insufficient for one of the Stage-2 criteria, mark it as uncertain and use BORDERLINE_REVIEW rather than pretending it is known.

20. Do not waste storage

This server has limited local storage.

Before large operations run:

df -h /workspace
du -sh /workspace/data/* 2>/dev/null | sort -h

For MM-AU especially:

metadata first
↓
filter IDs
↓
identify minimal files/shards
↓
download
↓
extract selected videos only
↓
delete temporary archives

Never:

download entire MM-AU first
then decide what is useful

unless there is technically no other way, and if so, explain the constraint before performing a destructive/huge operation.

21. Final outputs I expect

When finished, leave me with:

1. Filtered high-confidence Stage-2 video set
2. Separate borderline review set
3. Auto-rejection manifest with reasons
4. Download failure manifest
5. Deduplication report
6. Normalized metadata
7. Human-labeling manifest
8. Easy labeling interface
9. Persistent annotation output
10. Reproducible scripts/configs
11. R2 backup of valuable outputs
12. Final experiment/data report

The primary queue should contain videos that already satisfy, as strongly as metadata permits:

ego dashcam
+
actual collision
+
ego physically involved
+
ego maintaining its lane
+
another vehicle entering/crossing/merging into ego path
+
entry visible
+
collision visible

so that my job is mainly to label:

exact ENTRY frame
exact COLLISION frame
LEFT / RIGHT
evasion_space 0 / 1

rather than spending most of my time rejecting irrelevant videos.

22. Final report

Create:

reports/stage2_external_dataset_report.md

Include:

MM-AU total metadata records
CausalCrash total metadata records

number rejected at each filtering stage
number entering borderline review
number entering high-confidence queue

number of selected downloads
successful downloads
failed/unavailable downloads

duplicates found
final unique candidate count

MM-AU final count
CausalCrash final count

disk space used

filter rules actually used
fields actually available in each dataset
important deviations from this prompt
known uncertainties

exact paths to final videos/manifests
exact R2 paths
exact command to launch labeling

Also print a concise summary to the terminal when everything is complete.

23. Execution behavior

Do the work, not only write a plan.

Start by inspecting the current repository and disk/R2 state.

Implement the metadata-first pipeline.

Run it.

Inspect intermediate results for obvious mistakes.

If a filter unexpectedly keeps almost everything or almost nothing, investigate instead of blindly proceeding.

Manually inspect a small random sample from:

high-confidence
borderline
auto-rejected

to sanity-check the filtering logic before downloading a very large amount of data.

Keep a strong bias toward high precision in the primary labeling queue.

The end goal is:

When I open the labeling interface, most clips should already be genuine Stage-2-compatible examples, and I should mainly be assigning the four final Stage-2 labels rather than performing dataset discovery/filtering myself.