# RGB 교사 사용 기록

이번 교사는 DINOv2 ViT-B/14 with four registers 한 종류다. 공개 가중치를 고정했고 DEV train RGB 외 대회 데이터로 적응시키지 않았다.

- 공식 저장소: https://github.com/facebookresearch/dinov2
- 논문: https://arxiv.org/abs/2304.07193
- 정확한 source revision, checkpoint URL/SHA, LICENSE SHA: teacher_manifest.json.
- DINOv2 code/model의 Apache-2.0 조건을 확인했다. Cell-DINO/XRay-DINO를 사용한 것이 아니다.
- 로컬 참가 안내서는 학습 시 별도 공개 데이터셋 사용과 추가 사용 사항 제출을 설명한다. 이번에는 외부 데이터셋을 직접 내려받지 않고 공개 RGB 사전학습 모델을 교사로 사용했다. 최종 제출 모델에서 이를 채택하면 모델/가중치 출처를 추가 사용 사항으로 함께 명시한다.
- 공개 모델의 모든 사전학습 이미지와 대회 영상의 완전 무중복을 확인했다는 주장은 하지 않는다.
- 원시 status/goal/pose 또는 미래 GT trajectory는 teacher에 입력하지 않는다.
- Teacher 입력은 학생에게 제공된 동일 camera/crop/flip/photometric RGB이며, 별도 DINO augmentation을 추가하지 않는다.
- 교사의 spatial patch token만 이용한다. CLS 및 네 register token은 제외한다.
- 교사와 projection은 inference graph에서 제외한다. 이번 A2 FULL 후보는 기존 가중치라 시각 교사 실험과 무관하다.

검사 근거는 visual_preflight.json, 입력·계보는 INPUT_POLICY.json / CURRENT_BASE.json, 비용은 COST_CHECK.json에 있다. 기존 A2 입력 경계를 지켰다는 설명이며 운영국 개별 승인 문구를 대신하지 않는다.
