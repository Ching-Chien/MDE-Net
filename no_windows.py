import cv2
import os
import time
import torch
import numpy as np
from model import LDRN
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
import torch.quantization
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import imageio

all_block_averages = {'inner': [], 'middle': [], 'outer': []}

def divide_image_into_blocks(image, num_blocks_x, num_blocks_y):
    h, w = image.shape
    block_height = h // num_blocks_y
    block_width = w // num_blocks_x

    blocks = []
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block = image[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
            blocks.append(block)
    return blocks

def draw_grid_lines(ax, num_blocks_x, num_blocks_y, block_height, block_width):
    for i in range(1, num_blocks_y):
        y = i * block_height
        ax.axhline(y=y, color='red', linestyle='--', linewidth=0.5)
    for j in range(1, num_blocks_x):
        x = j * block_width
        ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5)
        
# 相機截圖的配置
def gstreamer_pipeline(sensor_id=0, capture_width=1920, capture_height=1080, display_width=960, display_height=540, framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

# 深度學習模型的配置
def setup_model():
    parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Directory setting 
    parser.add_argument('--model_dir',type=str, default = '')
    parser.add_argument('--img_dir', type=str, default = None)
    parser.add_argument('--img_folder_dir', type=str, default= None)

    # Dataloader setting
    parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

    # Model setting
    parser.add_argument('--encoder', type=str, default = "MobileNetV2")
    parser.add_argument('--pretrained', type=str, default = "KITTI")
    parser.add_argument('--norm', type=str, default = "BN")
    parser.add_argument('--n_Group', type=int, default = 32)
    parser.add_argument('--reduction', type=int, default = 16)
    parser.add_argument('--act', type=str, default = "ReLU")
    parser.add_argument('--max_depth', default=50.0, type=float, metavar='MaxVal', help='max value of depth')
    parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

    # GPU setting
    parser.add_argument('--cuda', action='store_true', default = "cuda")
    parser.add_argument('--gpu_num', type=str, default = "0,1,2,3", help='force available gpu index')
    parser.add_argument('--rank', type=int,   help='node rank for distributed training', default=0)

    args = parser.parse_args()

    assert args.model_dir != '', "Expected pretrained model directory"

    if args.cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
        cudnn.benchmark = True
        print('=> on CUDA')
    else:
        print('=> on CPU')

    if args.pretrained == 'KITTI':
        args.max_depth = 80.0
    elif args.pretrained == 'NYU':
        args.max_depth = 10.0

    print('=> loading model..')
    Model = LDRN(args)
    if args.cuda and torch.cuda.is_available():
        Model = Model.cuda()
    #Model = torch.nn.DataParallel(Model)
    Model.load_state_dict(torch.load(args.model_dir))
    Model.eval()

    # 對模型進行動態量化
    Model = torch.quantization.quantize_dynamic(Model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

    return Model, args

# 深度學習模型的處理過程
def process_image_with_model(model, args, frame, frame_count):
    # 將OpenCV的frame轉換為PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = np.asarray(img, dtype=np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
        img = np.repeat(img, 3, 2)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)
    if args.cuda and torch.cuda.is_available():
        img = img.cuda()

    _, org_h, org_w = img.shape

    # new height and width setting which can be divided by 16
    img = img.unsqueeze(0)

    if args.pretrained == 'KITTI':
        new_h = 352
        new_w = org_w * (352.0 / org_h)
        new_w = int((new_w // 16) * 16)
        img = F.interpolate(img, (new_h, new_w), mode='bilinear')
    elif args.pretrained == 'NYU':
        new_h = 432
        new_w = org_w * (432.0 / org_h)
        new_w = int((new_w // 16) * 16)
        img = F.interpolate(img, (new_h, new_w), mode='bilinear')

    img_flip = torch.flip(img, [3])
    with torch.no_grad():
        _, out = model(img)
        _, out_flip = model(img_flip)
        out_flip = torch.flip(out_flip, [3])
        out = 0.5 * (out + out_flip)

    if new_h > org_h:
        out = F.interpolate(out, (org_h, org_w), mode='bilinear')
    out = out[0, 0]

    if args.pretrained == 'KITTI':
        out = out[int(out.shape[0] * 0.15):, :]
        out = out * 256.0
    elif args.pretrained == 'NYU':
        out = out * 1000.0
    out = out.cpu().detach().numpy().astype(np.uint16)
    out = (out / out.max()) * 80.0
    test_filename = f'./test/out_{frame_count}.jpg'
    plt.imsave(test_filename, np.log10(out), cmap='plasma_r')
    blocks = divide_image_into_blocks(out, 5, 5)
    block_averages = [block.mean() for block in blocks]
    all_block_averages['inner'] = [block_averages[12]]
    all_block_averages['middle'] = [block_averages[i] for i in [6, 7, 8, 11, 13, 16, 17, 18]]
    all_block_averages['outer'] = [block_averages[i] for i in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]]
    print(all_block_averages['inner'])
    print(all_block_averages['middle'][3])
    print(all_block_averages['middle'][4])
    
    for i, block in enumerate(blocks):
        block[...] = block_averages[i]

    depth_image_modified = np.vstack([np.hstack(row) for row in np.array_split(blocks, 5)])
    result_filename = f'./processed_frames/out_{frame_count}.jpg'
    if not os.path.exists('./processed_frames'):
        os.makedirs('./processed_frames')
    plt.imsave(result_filename, np.log10(depth_image_modified), cmap='plasma_r')

    if args.cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空CUDA 缓存

    return result_filename,all_block_averages

# # 主功能
# def main():
#     Model, args = setup_model()

#     window_title = "CSI Camera"
#     start_time = time.time()
#     frame_count = 0

#     video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
#     if video_capture.isOpened():
#         try:
#             cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
#             while True:
#                 ret_val, frame = video_capture.read()
#                 current_time = time.time()
#                 if current_time - start_time >= 15:  # Save frame every 15 seconds
#                     frame_count += 1
#                     process_image_with_model(Model, args, frame, frame_count)
#                     start_time = time.time()  # Reset the start time

#                 if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
#                     cv2.imshow(window_title, frame)
#                 else:
#                     break
#                 keyCode = cv2.waitKey(10) & 0xFF
#                 if keyCode == 27 or keyCode == ord('q'):
#                     break
#         finally:
#             video_capture.release()
#             cv2.destroyAllWindows()
#     else:
#         print("Error: Unable to open camera")

def main():
    
    # drone = System()
    # await initialize_drone(drone)
    
    Model, args = setup_model()

    start_time = time.time()
    frame_count = 0

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            while True:
                ret_val, frame = video_capture.read()
                current_time = time.time()
                if current_time - start_time >= 10:  # Save frame every 15 seconds
                    frame_count += 1
                    result_filename, all_block_averages = process_image_with_model(Model, args, frame, frame_count)
                    # await avoid_obstacle_with_velocity_ned_yaw(drone, all_block_averages)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print("execution_time",execution_time,"s")
                    start_time = time.time()  # Reset the start time

                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    print("-- Landing")
                    # await drone.action.land()
                    break
        finally:
            print("-- Landing")
            video_capture.release()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    main()
