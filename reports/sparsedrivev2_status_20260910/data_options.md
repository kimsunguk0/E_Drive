현재 SDV2의 `PlanDataset`을 유지하면서 front의 −1·−5프레임을 추가할 수 있습니다. 기존 train/tune 전 행의 파일 존재와 보조 라벨 행 대응을 CPU로 확인했고, 최소 대조용 wrapper도 작성·검증했습니다. GPU·모델 추론·학습은 실행하지 않았습니다.

## 구현 및 입력 계약

로컬 구현은 `temporal_data.py`, 테스트는 `test_temporal_data.py`입니다.

- `temporal_data.py` SHA256: `4fc75a92b503d472a0fcb18189cbcedce6dd4c2c722219e04c0c0fa4d09f672b`
- `test_temporal_data.py` SHA256: `d2ced7e4d07e55005df61f7f3b4dccc77f6bb61abaf8e2594aa7ab83cbfcf0e0`
- frozen base `experiments/sparsedrivev2_20260910/data.py` SHA256: `54bcd88b1c2567d8aab3337fd82f5e6be50a4950e51957ba1df2908185f1eddb`

```python
base = PlanDataset(..., status_mode="zero", goal_mode="selection")
dataset = TemporalPlanDataset(base, history_mode="real", auxiliary=True)
inputs = model_inputs(batch)                       # A/B: no actual state
inputs_c = model_inputs(batch, common_status=True)  # C: perception_status only
```

`history_mode="repeat"`는 현재 front 영상 tensor를 두 번 복제합니다. `"real"`은 같은 scene의 현재 `frame−1`, `frame−5` JPEG를 읽습니다. 두 모드에서 현재 3카메라 영상, calibration, target, 행 순서는 동일합니다. image size는 양쪽 모두 512×256으로 고정했습니다. `history_images`는 item `[2,3,256,512]`, batch `[B,2,3,256,512]`, `time_offsets`는 nominal `[.1,.5]`초입니다. 과거 pose 정렬·raw timestamp·GT 변환행렬은 모델 입력에 없습니다.

기존 loader는 camera 순서 front-left/front/front-right에 따라 RNG에서 brightness/contrast/color를 각각 뽑습니다. wrapper는 같은 seed·epoch·row의 RNG에서 front에 사용된 세 계수만 재현해 과거 두 영상에 동일 적용합니다. 원 loader의 현재 영상은 다시 계산하거나 수정하지 않습니다. augment=True에서 source SHA가 바뀌면 RNG 재현을 가정하지 않고 거부합니다. 실제 legacy `PlanDataset`이 같은 RNG로 과거 frame을 읽었을 때와 wrapper의 history tensor가 bitwise 일치하는 CPU 통합 검증도 통과했습니다.

기본 `model_inputs`의 key는 `images,lidar2img,image_hw,history_images,time_offsets` 다섯 개뿐입니다. 원 `status`는 항상 8D zero이며 helper에서 제외합니다. 모든 arm의 `state_target/state_valid`는 아래 nominal causal overlay의 4D 상태와 all-true mask입니다. `causal_status4`는 별도 tensor로 보존하며, C에서 명시적으로 `common_status=True`일 때만 `perception_status`에 대응시킵니다. state_target을 바꾸어도 causal_status4가 바뀌지 않습니다. Goal은 outer sample에 둘 수 있지만 helper에 들어가지 않으며, 최종 후보 선택기 caller가 별도로 전달해야 합니다. shared-perception 모델이 실제로 이 경계를 지키는지는 해당 모델 구현의 별도 검증 대상입니다.

## 영상 및 행 대응 실측

원격 base는 `/NHNHOME/data/sukim/adcl`, 현재 worktree는 `experiment_worktrees/sparsedrivev2_20260910`입니다. 영상 경로는 다음과 같습니다.

```text
/NHNHOME/data/sukim/adcl/cache/etri_768/{scene}/camera_front/{frame:08d}.jpg
```

현재 3카메라 순서는 `camera_front_left,camera_front,camera_front_right`이며 front index=1입니다. 원 cache는 undistorted/cropped RGB 768×432입니다. 기존 PIL bilinear resize 및 ImageNet RGB mean/std를 그대로 사용합니다. 원 projection의 camera indices는 `(2,0,1)`이고, 가로 512/768·세로 256/432로 기존과 같이 rescale합니다.

| 모집단 | 현재 행 수 | 현재 frame 범위 | 필요한 −1/−5 front 파일 수(중복 제거) | 누락 |
|---|---:|---|---:|---:|
| train203 | 54,810 | 30…299 | 55,622 | 0 |
| tune37 | 1,998 | 30…295, stride5 | 3,996 | 0 |

파일 존재는 두 모집단 전체에 확인했습니다. 이미지 geometry/RGB header는 train/tune 처음·마지막 scene의 current/−1/−5 총12개를 읽어 모두 768×432 RGB임을 확인했습니다. 전 파일의 실제 JPEG decode 검증은 하지 않았습니다. 파일 선택은 global `row−lag`를 무조건 가정하지 않고, 반드시 `(현재 scene, 현재 frame−lag)`로 합니다. 원본 ego-cache의 global row는 라벨과 현재 sample의 identity로 유지합니다.

- split manifest: `data/etri/motiondrive_v2/grouped_split_rawtime.json`, SHA `f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936`
- ego cache: `/tmp/pm97/data/etri/ego_cache.npz`, SHA `d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd`
- train rows SHA: `75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854`
- tune rows SHA: `1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88`

## 현재 공통 인지 보조 라벨

재사용할 버전은 아래 geometry_v2입니다. 이전 `train_tune`은 split SHA가 다르고 `train_tune_rawtime`은 이전 calibration lineage이므로 같은 이름의 라벨이라고 임의 혼용하면 안 됩니다.

```text
/NHNHOME/data/sukim/adcl/data/etri/motiondrive_v2/train_tune_geometry_v2/
  supervision_manifest.json
  calibration.npz
  {scene}.npz
  {scene}.json
```

supervision manifest SHA는 `ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93`, 실제 canonical calibration SHA는 `8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961`입니다. manifest의 과거 pkl `calibration_sha256`와 현재 `canonical_calibration_sha256`를 혼동하지 않아야 합니다.

240개 scene의 row/frame 배열과 모든 target header를 확인했습니다. 각 scene은 frame30…299의 270행을 포함합니다. train54,810행과 tune1,998행은 모두 global row·scene·frame이 ego cache 및 split과 일치합니다. tune 캐시도 scene당270행이 있으므로, 평가에 사용할 54행은 반드시 `row` lookup으로 선택해야 합니다.

| 필드 | scene 단위 shape | dtype | 의미 |
|---|---|---|---|
| row / frame | `[270]` | int64 | global ego-cache row / scene-local frame |
| occ_target / lane_target | `[270,1,64,48]` | float32, binary | 현재 annotated object footprint / 실제 map polyline |
| occ_valid / lane_valid | `[270,1,64,48]` | bool | 감독 가능한 셀만 true |
| state_target / state_valid | `[270,6]` | float32 / bool | 기존 P7 raw-time state5+stop; 새 wrapper는 이 필드를 읽지 않음 |
| history_target / history_valid | `[270,4,4]` | float32 / bool | offsets1,2,5,10의 dx,dy,sin yaw,cos yaw; 새 wrapper는 읽지 않음 |
| history_transforms | `[270,4,4,4]` | float32 | GT pose 정렬 행렬; 새 wrapper는 읽지 않음 |

grid는 x `[-10,70]`m, y `[-32,32]`m, axis0가 forward x, axis1이 left y입니다. cell center는 `x=-10+(i+.5)*1.25`, `y=-32+(j+.5)*(64/48)`입니다. current3의 z=0 또는1m projection이 positive depth>0.05, u∈[0,W), v∈[0,H)에 드는 셀과 기존 valid mask를 교차합니다. 실제 full6 visibility는3,064셀, current3 visibility는2,677셀로387셀이 제외됩니다. 이는 pose 정렬 없이 고정 calibration만으로 계산합니다.

Occupancy는 `num_points>0`인 현재 non-ego annotated footprint입니다. 음성은 유효 객체 중심 bounding rectangle+3m, camera visibility, radius≤40m 등의 제한된 지지 영역입니다. 객체가 없거나 invalid하거나 가려졌다고 물리적 free space로 간주하지 않습니다. Lane도 실제 map polyline 주변의 제한된 support이며 모든 주행가능영역을 뜻하지 않습니다. raw object의 width가 heading축 길이, length가 횡방향 길이인 원 converter 계약도 기존 라벨에 반영되어 있습니다.

기존 `motiondrive_v2_training.py::balanced_raster_bce`의 label/mask 처리와 foreground/background 균형은 재사용할 수 있습니다. target 또는 task의 valid가 batch 전체에서0인 경우는 정상적인 unknown이므로 해당 task loss를0으로 처리해야 합니다. 무조건 오류로 종료하거나 unknown을 음성으로 바꾸면 안 됩니다. 전체 train/tune의 mask 값 분포는 이번에 전수 집계하지 않았습니다.

wrapper는 aux NPZ에서 row/frame/occ/lane 네 raster만 읽습니다. constructor에서 split/ego/calibration SHA를 한 번 검증한 후 고정값으로 재사용합니다. scene cache는 worker당 LRU256이며 binary targets를 bool로 압축 보관하고 item 반환에서만 float32로 변환합니다. 실제 scene당 raster memory는3,317,760bytes(약3.16MiB), train203 전체 약643MiB/worker, 4worker 약2.51GiB/arm입니다. 원격 RAM은 총 약2.2TiB, available 약2.1TiB로 확인했습니다. 반복 scene miss마다45MB ego cache를 해시하거나 작은 LRU로 NPZ를 계속 다시 풀지 않습니다.

## 새로운 상태 보조 라벨과 actor motion의 범위

이번 wrapper의 state_target은 P7의 raw-time state_target을 사용하지 않습니다. 모든 arm에서 다음 frozen nominal causal overlay의 first4를 사용합니다.

```text
/NHNHOME/data/sukim/adcl/data/etri/motiondrive_v2_shared_status_a1_20260908_ops/
  overlay_manifest.json
  train.npz / tune.npz
```

source `status5`는 current ego `vx,vy,ax,ay,yaw_rate`; 새 label은 처음4개만이며 m/s,m/s²입니다. 정확히 과거−10…0의11개 pose, nominal10Hz quadratic fit으로 계산했고 future values는 사용하지 않습니다. row/frame과 artifact SHA를 검증하여 현재 subset만 읽습니다. P7 raw-time 감독값과 정의가 다르다는 점은 `empirical/EMPIRICAL_STATUS_KO.md`에 수치와 함께 기록했습니다. 기존 state smooth-L1을 재사용하면 scale first4는 `(10,5,3,3)`이며, 새 trainer의 실제 선택한 normalization을 명시해야 합니다. C에서 입력으로 쓰는 상태도 이 동일 source이며 `perception_status`라는 명시적 opt-in 경로로만 전달됩니다.

기존 history target을 추가 보조학습에 쓰기로 한다면 offsets1·5는 원 `[1,2,5,10]`의 index0·2입니다. dx/dy는 metre, sin/cos는 무차원, valid는 각 성분 bool, 기존 normalization은 `(10,5,1,1)`입니다. 현재 wrapper에는 이를 구현하지 않았으며, label용 relative pose와 inference image alignment를 혼동하면 안 됩니다.

확인한 geometry_v2 NPZ key에는 actor identity·actor motion·future occupancy가 없습니다. 원본 `data/etri/meta_train/{scene}/annotation/object.parquet`는 train/tune240scene 모두 존재합니다. 표본3scene에서 `timestamp,class,obj_id,x[m],y[m],z[m],heading[rad],width[m],length[m],height[m],score,vx,vy,num_points`를 확인했습니다. 따라서 새 actor supervision을 만들 재료는 있지만, obj_id의 시간적 일관성·scene 간 수명·중복·velocity 좌표계·dropout mask를 검증하지 않았으므로 곧바로 재사용 가능한 motion label로 주장할 수 없습니다. 추진한다면 scene 안에서 timestamp/frame을 정확히 결합하고 track ID와 유효 박스를 검증한 별도 target 생성이 필요합니다. 이는 현재 최소 대조에 포함하지 않았습니다.

## 검증 결과와 남은 범위

CPU synthetic 8개 테스트 PASS: repeat tensor 독립성과 현재영상 보존, 정확한 과거 frame 선택 및 누락 실패, 기본/조건부 입력 경계, GT state alias 분리, unknown/시야 mask 보존, causal future/hash 위반 거부, raw planner status 거부, aux row/frame mismatch 거부, per-row 해시 제거와 compact cache를 확인했습니다.

실제 원격 데이터 CPU 통합도 PASS했습니다. train rows `[30,111299]`, tune `[14730,111595]`, augmentation on/epoch3에서 현재 영상과 모든 공통 label이 두 history 모드 사이에 bitwise 동일합니다. real history는 frozen PlanDataset의 같은 RNG·과거 frame 전처리와 bitwise 동일합니다. 최종 source SHA `4fc75a…`에서 aux cache 수정 후 재검증했고, per-item hash 호출을 강제로 오류 처리해도 실제 네 행이 통과했습니다. CUDA는 초기화하지 않았고 원격 소스 파일은 작성하지 않았습니다.

완료된 것은 데이터 준비와 CPU 계약 검증입니다. 새 architecture 학습 성능·GPU 처리량·전체 latency·공통 perception 조건의 실제 모델 경로는 아직 이 자료로 확인되지 않았습니다. 이후 subgroup 학습은 원 `PlanDataset`의 explicit rows 제한을 유지해야 하며, cache가 존재한다는 사실이 해당 run의 학습 행 사용 허가를 확대하지는 않습니다.
