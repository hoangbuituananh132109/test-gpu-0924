# 8xB200 offline runner

Gói này chạy một node 8 GPU, không tải model/dataset từ mạng và mặc định dùng:

- student: `/shared-storage/models/Qwen3-1.7B` (Qwen3-1.7B-Instruct);
- teacher: `/shared-storage/models/Qwen3-4B-Base-GRPO`;
- train data: `/shared-storage/models/dapo_processed_pilot_512.parquet`;
- checkpoint/log: `/shared-storage/models/opd-grpo-frontier-runs`;
- code clone/extract dưới `/nlp/anhhbt/Rethinking-OPD`.

Tất cả đường dẫn đều có thể override bằng biến môi trường. Preflight bắt buộc
đủ 8 B200, đúng 512 dòng và đủ file model trước khi tạo Ray workers.

Profile 8 GPU dùng `TRAIN_BATCH_SIZE=16` và `ppo_mini_batch_size=16` để mỗi
optimizer minibatch chia đều trên 8 rank. Đây là một regime throughput mới;
muốn so sánh khoa học thì phải chạy lại mọi nhánh cần đối chiếu (ít nhất E1,
E4 và E5) cùng profile này. Không so trực tiếp điểm của nó với run Modal/Vast
batch 2 rồi quy khác biệt cho queue/controller.

## Cài code offline

```bash
cd /nlp/anhhbt/Rethinking-OPD
PYTHONPATH="$PWD/verl:$PWD" python3 -c 'import verl.trainer.main_ppo, frontier' || \
  python3 -m pip install -e ./verl --no-deps
```

Chỉ dùng `pip` khi environment đang thiếu package. Runner đặt toàn bộ Hugging
Face/Transformers/Datasets và telemetry ở chế độ offline.

## Preflight và smoke 2 step

```bash
cd /nlp/anhhbt/Rethinking-OPD
bash frontier_company/preflight_b200.sh frontier_company/config_b200_8gpu.env
bash frontier_company/run_b200.sh frontier_company/config_b200_8gpu_smoke.env E5_hybrid_dynamic_queue
```

Smoke dùng response ceiling 1024 nên chỉ xác nhận pipeline, OOM và logging.
Không dùng accuracy của smoke làm kết quả nghiên cứu.

## Chạy E5 7k nền

```bash
cd /nlp/anhhbt/Rethinking-OPD
bash frontier_company/start_b200_background.sh frontier_company/config_b200_8gpu.env E5_hybrid_dynamic_queue
```

Theo dõi hoặc dừng đúng process group:

```bash
bash frontier_company/status_b200.sh frontier_company/config_b200_8gpu.env E5_hybrid_dynamic_queue
bash frontier_company/stop_b200.sh frontier_company/config_b200_8gpu.env E5_hybrid_dynamic_queue
```

Để chạy nhánh khác, thay experiment ID bằng `E1_grpo_base`, `E2_opd_base`,
`E3_hybrid_static` hoặc `E4_hybrid_dynamic`. Giữ nguyên data, G, response
ceiling và optimizer budget khi dùng chúng cho so sánh.

## Attention backend

`ATTN_IMPLEMENTATION=auto` dùng FlashAttention2 + remove-padding khi package
`flash_attn` đã có. Nếu không có, nó dùng PyTorch SDPA và tắt remove-padding;
không tự cài và không truy cập mạng. Log luôn ghi backend thực tế.

## Phạm vi bằng chứng

Theo policy hiện tại của dự án, compute công ty dùng cho engineering/smoke và
chẩn đoán. Không trộn số liệu này vào bảng paper vốn yêu cầu compute cá nhân.
