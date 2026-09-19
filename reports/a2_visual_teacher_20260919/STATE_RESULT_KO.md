# QREFINE 예측 state/history 정보 ON/OFF

동일 고정 DEV 가중치의 추론 개입이다. 제공 status와 continuous motion 입력은 유지했다.

|조건|PREFIX|L2_1s|L2_2s|L2_3s|일반 주행|
|---|---:|---:|---:|---:|---:|
|ON|0.164251762|0.096046498|0.163684602|0.233024186|0.168445111|
|OFF|0.164534473|0.096314961|0.164016025|0.233272433|0.168745465|

OFF−ON: +0.000282711. Session95%CI: [-0.0008153227963598122, 0.0013362275999347763].
두 계획의 PREFIX 거리: 0.007766543.
Scene/motion/state/history 출력은 ON/OFF 간 동일했고 parameter/buffer hash와 원래 flag를 확인했다.
OFF는 MLP 입력 정보를 0으로 만든다. Bias로 생기는 constant token은 남는다.
이 결과는 학습부터 OFF인 모델의 성능을 의미하지 않는다. 자동 OFF 학습이나 제거 조합 스윕은 실행하지 않는다.
