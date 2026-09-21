# Backbone capacity / full decoder split / agent future matched continuations

DEV 진행 중. 현재 평가를 terminal 성과로 해석하지 않는다.
부모 DEV 0.144597662 / 제출 FULL 0.133684828. 서로 다른 평가이며 환산하지 않는다.

|Stage step|CTRL|R101|DECSPLIT|AGENT|
|---|---:|---:|---:|---:|
|0|0.144597662|0.144597662|0.144597662|0.144597662|

같은 sample 순서 146 update 확인.
각 treatment−CTRL이 주 비교다. 부모 대비는 추가 학습 효과를 포함한다.
시점·일반주행·종횡·구간 벡터·인지·세션 paired CI는 result_step*.json에 보존한다.
FULL 및 공식 업로드는 실행하지 않는다.
