@echo off
cd /d "%~dp0\.."

rem Phase A: 751 labeled sensor times and 1501 unlabeled physics points.
rem No checkpoint trained on 1501 sensor labels is loaded.
python src\mvm_pinn_jera1.py --stiffness-profile sigmoid --relative-formulation first_order --first-order-physics weak --sigmoid-robustness sparse751_densephysics --first-order-pretrain-only --pretrain-epochs 6500 --noise 0 --measurements 751 --outdir outputs\first_order_weak_sigmoid_sparse751_densephysics
if errorlevel 1 exit /b %errorlevel%

rem Phase B: joint RelativeStateNet + sigmoid stiffness optimization.
python src\mvm_pinn_jera1.py --stiffness-profile sigmoid --relative-formulation first_order --first-order-physics weak --sigmoid-robustness sparse751_densephysics --relative-state-checkpoint outputs\first_order_weak_sigmoid_sparse751_densephysics\relative_state_sparse751_densephysics.pt --epochs 6000 --noise 0 --measurements 751 --sigmoid-seeds 2026 2027 2028 --outdir outputs\first_order_weak_sigmoid_sparse751_densephysics
