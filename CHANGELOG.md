# CHANGELOG

### This file contains changes and modifications during developing.

## Jan 21, 2021, 14:27
- Thay đổi cách xử lí `bert-path` trong file `configs.py` - nếu kiểm tra không thấy 
- Thêm phần kiểm tra multi GPU để dùng `DataParallel` trong `para_selector.py` 


## Jan 20, 2021, 14:50
- Thay đổi lại nơi lưu các file tạo ra trong quá trình chạy từ `./backup_files` sang folder lưu data chung `_data/QA/HotpotQA/backup_files`. Việc này nhằm đưa data (cả dataset và các file trong quá trình xử lí) ra khỏi code folder.
- Thay đổi optimizer của Paras Selector từ optimizer chuẩn của `torch` sang `AdamW` của `huggingface`
- Thêm `scheduler` để thay đổi learning rate của `AdamW`
- Thay đổi để bây giờ code sẽ được clone về từ git thay vì extract từ file `tar` trên `Isilon`


## Jan 6 2021, 10:01 AM
- Thêm 'local2' là máy laptop ASUS vào `args.working_place`
- Thêm class hỗ trợ xử lí song song `ParallelHelper` vào `modules\utils.py` và thay đổi lại cách xử lý song song trong `modules/para_selection/DataHelper.py`
- Thay đổi cấu trúc của object mà được pickle vào các file `backup_files/select_paras/dataset_(train/test/dev).pkl.gz`