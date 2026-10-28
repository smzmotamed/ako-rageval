<div dir="rtl">

# ako-rageval

[English](README.md)

**ابزار ارزیابی خط لوله‌های RAG فارسی.** chunking، retriever و مدل‌های embedding را روی پیکره‌ی (corpus) خودتان مقایسه کنید، کیفیت پاسخ را ردگیری کنید و اگر کیفیت بازیابی افت کرد، CI را شکست بدهید.

متن فارسی مشکلاتی دارد که ابزارهای عمومی RAG نادیده می‌گیرند: `ي`/`ک` عربی و فارسی، ارقام فارسی، عربی و لاتین، نیم‌فاصله (ZWNJ در `می‌روم` / `می روم` / `میروم`)، اعراب، `؟` و `«»`. این ابزار آن‌ها را برای تطبیق یکسان می‌کند، بازیابی را در **سطح سند** می‌سنجد تا chunkerها قابل مقایسه باشند، و گزارش HTML مستقل (انگلیسی یا فارسی راست‌به‌چپ) می‌سازد.

- **بدون وابستگی زمان اجرا**، Python نسخه‌ی ۳٫۱۰ به بالا، BM25 خالص.
- **نرمال‌سازی فارسی** با سیاست مشخص برای نیم‌فاصله و ارقام.
- **chunkerها:** `doc`، `paragraph`، `sentence` (جداکننده‌ی جمله‌ی فارسی) و `fixed`.
- **retrieverها:** BM25 روی واژه، BM25 روی n-gram نویسه (مقاوم در برابر ریخت‌شناسی)، حالت بدون نرمال‌سازی برای مقایسه، **retriever برداری برای هر تابع embedding**، ترکیب رتبه (RRF) و `module:factory` دلخواه.
- **متریک‌ها:** hit@k، recall@k، precision@k، nDCG@k و MRR؛ برای پاسخ `token_f1`، `token_recall` و `groundedness`؛ امکان اتصال LLM judge.
- **دروازه‌ی CI:** `--fail-under recall@5=0.9` در صورت افت کیفیت با کد خروج غیرصفر تمام می‌شود.

## شروع سریع

```bash
pip install ako-rageval

ako-rageval compare --sample
ako-rageval compare --sample --html report.html --report-lang fa
ako-rageval eval-answers --sample
```

> داده‌ی نمونه‌ی همراه ابزار کوچک و ساده است و فقط برای نمایش و smoke test است، **نه benchmark**. پیش از نتیجه‌گیری، یک دیتاست طلایی (golden dataset) از اسناد و پرسش‌های واقعی کاربرانتان بسازید.

## داده‌ی شما

سه فایل JSON Lines با کدگذاری UTF-8 (خطاها شماره‌ی فایل و خط را نشان می‌دهند):

```jsonc
// corpus.jsonl
{"id": "d01", "title": "دماوند", "text": "قله‌ی دماوند با ارتفاع ..."}
// dataset.jsonl: relevant شناسه‌ی همه‌ی اسنادی است که به پرسش پاسخ می‌دهند
{"id": "q01", "question": "بلندترین قله‌ی ایران کدام است؟", "relevant": ["d01"], "reference_answer": "دماوند"}
// predictions.jsonl: خروجی سامانه‌ی RAG خودتان؛ contexts متن‌های بازیابی‌شده است
{"id": "q01", "answer": "دماوند", "contexts": ["قله‌ی دماوند با ارتفاع ..."]}
```

```bash
ako-rageval run --corpus corpus.jsonl --dataset dataset.jsonl --chunker sentence:3 --retriever bm25
ako-rageval compare --corpus corpus.jsonl --dataset dataset.jsonl \
    --chunkers doc,sentence:2,sentence:3,fixed:100:20 --retrievers bm25,bm25-char3 --html report.html
ako-rageval eval-answers --dataset dataset.jsonl --predictions predictions.jsonl --fail-under groundedness=0.8
```

کد خروج: `0` موفق، `1` آستانه‌ی `--fail-under` رعایت نشده، `2` ورودی یا استفاده‌ی نادرست.

## مقایسه‌ی مدل‌های embedding

رابط خط فرمان نمی‌تواند به سرویس embedding شما وصل شود؛ از API پایتون استفاده کنید. هر تابعی با شکل `embed(list[str]) -> list[vector]` کار می‌کند. نمونه‌ی کامل در [README.md](README.md) آمده است.

## درباره‌ی معیارهای پاسخ

`token_f1`، `token_recall` و `groundedness` معیارهای **واژگانی** و پایه‌اند: قطعی و ارزان‌اند و برای ردگیری افت کیفیت مناسب‌اند، اما معنا را نمی‌فهمند.

- `token_f1` پاسخ‌های طولانی را جریمه می‌کند؛ وقتی پاسخ مرجع کوتاه است از `token_recall` استفاده کنید.
- `groundedness` سهم واژه‌های پاسخ است که در contextهای بازیابی‌شده آمده. مقدار پایین نشانه‌ی احتمالی توهم (hallucination) است، اما مقدار بالا درستی پاسخ را ثابت نمی‌کند.

برای قضاوت معنایی، پروتکل `Judge` را پیاده‌سازی کنید و به `evaluate_answers(..., judge=...)` بدهید.

## نرمال‌سازی

حالت پیش‌فرض: NFKC، تبدیل حروف عربی به فارسی (`ي ى ك ة ۀ أ إ ٱ`)، حذف اعراب و کشیده، حذف نویسه‌های نامرئی، تبدیل نیم‌فاصله به فاصله، تبدیل ارقام به ASCII، تبدیل نقطه‌گذاری عربی، حروف کوچک لاتین و جمع‌کردن فاصله‌ها. یکی‌کردن `بزرگترین` و `بزرگ‌ترین` انجام نمی‌شود؛ BM25 روی n-gram نویسه (`bm25-char3`) راه‌حل ارزان است و جدول مقایسه نشان می‌دهد روی داده‌ی شما کمک می‌کند یا نه.

## مجوز

[Apache-2.0](LICENSE)

</div>
