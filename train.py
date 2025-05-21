from utils.dataset import dataset
from utils.common import PSNR
from model import FSRCNN
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=300000,          help='-')
parser.add_argument("--scale",          type=int, default=3,               help='-')
parser.add_argument("--batch-size",     type=int, default=128,             help='-')
parser.add_argument("--save-every",     type=int, default=100,             help='-')
parser.add_argument("--save-best-only", type=int, default=0,               help='-')
parser.add_argument("--save-log",       type=int, default=0,               help='-')
parser.add_argument("--ckpt-dir",       type=str, default="checkpoint/x3", help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAG, unparsed = parser.parse_known_args()
steps = FLAG.steps
batch_size = FLAG.batch_size
save_every = FLAG.save_every
save_log = (FLAG.save_log == 1)
save_best_only = (FLAG.save_best_only == 1)

scale = FLAG.scale
if scale != 3:
    raise ValueError("Only scale=3 is supported in this version.")

ckpt_dir = FLAG.ckpt_dir
if (ckpt_dir == "") or (ckpt_dir == "default"):
    ckpt_dir = f"checkpoint/x{scale}"

model_path = os.path.join(ckpt_dir, f"FSRCNN-x{scale}.pt")
#최종 학습이 완료된 후 저장할 FSRCNN 모델의 파일 경로
ckpt_path = os.path.join(ckpt_dir, f"ckpt.pt")
#학습 중간에 저장되는 체크포인트(중간 상태) 파일 경로



# -----------------------------------------------------------
#  Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"
lr_crop_size = 10
hr_crop_size = lr_crop_size * scale

train_set = dataset(dataset_dir, "train")
train_set.generate(lr_crop_size, hr_crop_size)
train_set.load_data()

valid_set = dataset(dataset_dir, "validation")
valid_set.generate(lr_crop_size, hr_crop_size)
valid_set.load_data()

#① 원본 이미지를 로딩, HR 이미지(train/image1.png)를 메모리로 읽음
#② 여러 개의 crop 위치를 랜덤으로 정함, 예를 들어 512x512 이미지에서 랜덤한 위치를 선택해 30x30짜리 crop을 뽑음
#③ HR crop과 그에 대응되는 LR crop을 생성, HR crop: 원본의 30x30 영역 그대로 사용. LR crop: HR crop을 1/3로 다운샘플 (→ 10x10로 만들기).
#④ 이런 쌍을 계속 만들어 메모리에 저장, 1개의 원본 이미지로부터 수십~수백 개의 (LR, HR) 쌍을 생성

#왜 crop을 쓰나?
#학습 데이터 수를 늘리기 위해: 원본 한 장으로도 수백 개의 학습 쌍 생성 가능
#메모리 효율성: 큰 이미지를 통째로 GPU에 넣는 것보다 작은 crop이 더 효율적
#다양한 위치와 패턴 학습 가능: 랜덤하게 자르면 다양한 텍스처를 모델이 배울 수 있음

# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsrcnn = FSRCNN(scale, device)
    fsrcnn.setup(optimizer=torch.optim.Adam(fsrcnn.model.parameters(), lr=1e-3),
                loss=torch.nn.MSELoss(), model_path=model_path,
                ckpt_path=ckpt_path, metric=PSNR)

    fsrcnn.load_checkpoint(ckpt_path)
    fsrcnn.train(train_set, valid_set, steps=steps, batch_size=batch_size,
                save_best_only=save_best_only, save_every=save_every,
                save_log=save_log, log_dir=ckpt_dir)

if __name__ == "__main__":
    main()
