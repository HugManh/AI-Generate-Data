# input-data
 
Tối ưu bộ nhớ với Generator:
Hàm process_single_file giờ trả về một Generator thay vì danh sách List[Tuple[str, str]]. Nó yield từng chunk ngay khi xử lý, không lưu toàn bộ vào RAM.
Trong preprocess_input_data, bỏ đoạn code tạo danh sách trung gian results và data_records. Thay vào đó, data_generator kết hợp multiprocessing và generator để stream dữ liệu trực tiếp.
Giữ multiprocessing:
Vẫn sử dụng Pool để xử lý song song các file, nhưng giờ mỗi file trả về generator thay vì list, nên không có overhead bộ nhớ lớn.
Streaming vào DataFrame:
Polars nhận dữ liệu từ data_generator thông qua tqdm, chỉ giữ một chunk trong bộ nhớ tại một thời điểm. Điều này phù hợp với file lớn.
Giới hạn num_rows:
Logic kiểm tra num_rows được đưa vào generator, dừng yield khi đủ số hàng, tránh xử lý dư thừa.