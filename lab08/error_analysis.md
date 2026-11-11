# Bài 20 — Error Analysis

## Thiết lập

- Model: `bert-base-uncased`, fine-tuned trên 1,000 review IMDb cân bằng, 1 epoch.
- Tập kiểm tra: 500 review IMDb cân bằng, cùng tập và thứ tự đã dùng ở Bài 19; seed = 43.
- Quy ước nhãn: `0 = negative`, `1 = positive`.
- Confidence là xác suất softmax lớn nhất của nhãn model dự đoán.
- Bài 19 có 87 lỗi trên 500 review: 18 false positives và 69 false negatives.

Các trích đoạn dưới đây giúp định vị ví dụ; cần xem review đầy đủ trong dataset nếu phần bị lược có thể đổi sắc thái. Ví dụ thứ tự 7–8 là hai lỗi sai confidence cao nhất còn lại sau khi đã chọn sáu ví dụ đầu, nên không bị trùng.

## False positives

True = negative, Pred = positive.

| Index | Text (excerpt) | True | Pred | Confidence | Error type | Hypothesis |
|---:|---|---|---|---:|---|---|
| 314 | “Some laughs are to be had … but the idea that young women would see anything desirable in … Stan or … Jack is just risible. Harmless enough in its time, now it's just a curio from a bygone age.” | negative | positive | 0.8832 | Sentiment hỗn hợp; ngữ cảnh dài | Review có kỷ niệm và cụm tích cực như “some laughs”, nhưng kết luận đánh giá các nhân vật là đáng chê và phim chỉ còn là món đồ cũ. Model có thể đã dựa nhiều vào tín hiệu tích cực ở phần đầu thay vì kết luận chê bai ở cuối. |
| 124 | “Judge and Jury begins with Joseph Meeker … There is an amusing scene … But once executed, Meeker returns as a revenant … to pay Silvano back … Director John Eyres … focusing solely on action scenes … that is the film's main flaw … The acting is surprisingly good …” | negative | positive | 0.8436 | Sentiment hỗn hợp; phụ thuộc ngữ cảnh | Review khen phim đầu của đạo diễn, một số cảnh và diễn xuất, nhưng đánh giá tổng thể nêu rõ kịch bản/action rời rạc và nhân vật thiếu chiều sâu. Các lời khen rải rác có thể lấn át nhận xét tiêu cực tổng thể. |
| 223 | “Deep Sea 3D is a stunning insight … It is a beautifully captured under sea feast for the eyes. Our ears … are not given the same treatment … The script is fairly silly … with its childish script, annoying narration and misplaced sound effects it cannot be taken seriously.” | negative | positive | 0.8429 | Đảo chiều đánh giá; sentiment theo khía cạnh | Phần đầu khen mạnh hình ảnh và phần nhạc, sau đó chuyển sang chê lời dẫn, kịch bản và hiệu ứng âm thanh; câu kết phủ định giá trị tổng thể. Model có thể đã bắt tín hiệu khen về hình ảnh mà không tổng hợp được phần chê và kết luận ở cuối. |

## False negatives

True = positive, Pred = negative.

| Index | Text (excerpt) | True | Pred | Confidence | Error type | Hypothesis |
|---:|---|---|---|---:|---|---|
| 160 | “Jack Frost 2 is neither [a B-grade horror film nor a ‘So Bad It's Good’ movie]. It is … a very good movie cleverly hidden inside a very bad one … a screwball comedy … cleverly written parody … As a horror movie … rates a zero. But … IT'S NOT A HORROR MOVIE.” | positive | negative | 0.9004 | Phủ định; sentiment phụ thuộc mục tiêu/khía cạnh | Các từ “bad”, “horrible” và “rates a zero” xuất hiện khi người viết bác bỏ cách xem phim như horror, không phải đánh giá phủ định toàn bộ phim. Review khen phim như screwball comedy/parody; model có thể đã gắn các từ tiêu cực với nhãn negative mà không theo dõi phạm vi phủ định và mục tiêu đánh giá. |
| 361 | “John Carpenter's ‘The Thing’ is undoubtedly one of the best horror movies ever made. Sadly … it is also one of the most underrated movies … It gives the viewer a lot more reasons to watch it … A masterpiece of terror that will never be forgotten.” | positive | negative | 0.8980 | Từ cảm xúc phụ thuộc ngữ cảnh; sentiment hỗn hợp | Review có nhiều từ mang sắc thái tiêu cực (“Sadly”, “panned”, “ignorance”, “depressing”, “chaos”, “terror”) khi nói về phản ứng của giới phê bình hoặc không khí phim. Quan điểm thực của người viết là khen phim xuất sắc và đáng xem; model có thể nhầm từ vựng về chủ đề kinh dị/phê bình với sentiment của tác giả. |
| 399 | “Sure it is no ‘Human Traffic’ or ‘Lock, Stock...’ but is by no means a flop … it shouldn't [be ignored] … sets out to … entertain us … This film is well worth seeing.” | positive | negative | 0.8966 | Phủ định và cấu trúc tương phản | Review nhắc đến “bad reviews”, “no …” và “ignored”, nhưng dùng chúng để bác bỏ ý kiến phim thất bại; kết luận là phim giải trí và đáng xem. Model có thể chưa xử lý tốt phạm vi của phủ định và chiều chuyển ý sau “but”. |

## Hai lỗi sai có confidence cao khác

| Index | Text (excerpt) | True | Pred | Confidence | Error type | Hypothesis |
|---:|---|---|---|---:|---|---|
| 445 | “A lot of people call Corbucci's films ‘depressing’. I find that … ‘bleak and unforgiving’ are more apt … Those minor gripes aside there's more than enough here … a very talented director …” | positive | negative | 0.8940 | Từ cảm xúc phụ thuộc ngữ cảnh; tương phản | Người viết chấp nhận phim có không khí “bleak and unforgiving” và kể một số điểm trừ, nhưng đó là mô tả phong cách chứ không phải kết luận ghét phim. Sau cụm “Those minor gripes aside”, review chuyển sang khen dứt khoát đạo diễn và phong cách. Model có thể dựa vào nhiều từ tiêu cực mà bỏ lỡ lập trường khen tổng thể. |
| 403 | “It's hard to find a bad Hitchcock movie … and SABOTEUR proves the point … I wrongly assumed the story … might not hold my interest. Was I wrong! The suspenseful plot gets cooking right off the bat … fun and clever …” | positive | negative | 0.8918 | Đảo chiều lập trường; ngữ cảnh dài | Mở đầu cố ý tạo kỳ vọng rằng phim có thể dở, sau đó “Was I wrong!” đảo chiều và phần còn lại khen nhịp phim, sự thú vị cùng dàn diễn viên. Model có thể đã dựa vào các từ tiêu cực ở phần mở đầu mà không cập nhật sentiment sau tín hiệu đảo chiều và kết luận tích cực. |

## Observation → evidence → interpretation

- **Observation:** Trên tập test cân bằng 500 review, model có 18 FP nhưng 69 FN; Recall positive = 0.7240 thấp hơn Precision positive = 0.9095.
- **Evidence:** Sáu FN được chọn có confidence sai từ 0.8966 đến 0.9004; các trích đoạn chứa từ tiêu cực hoặc nhắc ý kiến tiêu cực trước khi thể hiện lập trường khen phim. Ba FP có lời khen cụ thể nhưng kết luận tổng thể nghiêng về chê.
- **Interpretation (hypothesis):** Các ví dụ gợi ý model có thể dựa mạnh vào từ vựng cảm xúc cục bộ và khó tổng hợp sentiment có tương phản/đảo chiều trong review dài. Confidence cao cho thấy softmax có thể rất chắc với dự đoán sai; nó không chứng minh model hiểu đúng lập trường tác giả. Đây là giả thuyết từ 8 ví dụ, không phải bằng chứng nhân quả; cần kiểm tra thêm review và thí nghiệm có kiểm soát để xác nhận.
