@echo off
cd /d "%~dp0\.."

python src\mvm_pinn_jera1.py --stiffness-profile sigmoid --relative-formulation first_order --first-order-physics weak --sigmoid-robustness noise003 --first-order-pretrain-only --pretrain-epochs 6500 --noise 0.003 --measurements 1501 --noise-seed 3030 --outdir outputs\first_order_weak_sigmoid_noise003
if errorlevel 1 exit /b %errorlevel%

python src\mvm_pinn_jera1.py --stiffness-profile sigmoid --relative-formulation first_order --first-order-physics weak --sigmoid-robustness noise003 --relative-state-checkpoint outputs\first_order_weak_sigmoid_noise003\relative_state_pretrained_sigmoid_noise003.pt --epochs 6000 --noise 0.003 --measurements 1501 --noise-seed 3030 --sigmoid-seeds 2026 2027 2028 --outdir outputs\first_order_weak_sigmoid_noise003
