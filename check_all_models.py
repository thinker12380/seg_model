"""验证所有已注册模型能否正常构建和前向传播"""
import torch
from models import build_model, MODEL_REGISTRY
import traceback
import sys

NUM_CLASSES = 4
BATCH_SIZE = 1
H, W = 512, 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_lines = []
log_lines.append(f"Device: {device}")
log_lines.append("=" * 50)

for name in sorted(MODEL_REGISTRY.keys()):
    log_lines.append(f"\n验证模型: {name}")
    try:
        model = build_model(name=name, num_classes=NUM_CLASSES)
        model = model.to(device)
        model.eval()

        x = torch.randn(BATCH_SIZE, 3, H, W).to(device)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (BATCH_SIZE, NUM_CLASSES, H, W), f"输出形状异常: {out.shape}"

        # 简单统计参数量
        params = sum(p.numel() for p in model.parameters())
        log_lines.append(f"  [OK] 输出形状: {out.shape}, 参数量: {params/1e6:.2f}M")
    except Exception as e:
        log_lines.append(f"  [FAIL] {e}")
        log_lines.append(traceback.format_exc())

log_lines.append("\n" + "=" * 50)
log_lines.append("验证完成")

result = "\n".join(log_lines)
print(result)

with open("model_check.log", "w", encoding="utf-8") as f:
    f.write(result)
