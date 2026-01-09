إليك محتوى مقترح لملف "اقرأني" (README.md) بشكل احترافي ومنظم باللغتين العربية والإنجليزية، ليناسب نشره على منصات مثل GitHub أو مشاركته مع زملائك.

📊 تطبيق اختبار وجود خطأ القياس (Measurement Error Test App)

تطبيق تفاعلي تعليمي وتحليلي مصمم لشرح وتبسيط منهجية اختبار وجود خطأ القياس في المتغيرات التفسيرية، بناءً على الأوراق العلمية لـ Wilhelm (2018) و Lee & Wilhelm (2019).

📝 وصف المشروع

يهدف هذا التطبيق إلى مساعدة الباحثين وطلاب الدراسات العليا في الاقتصاد القياسي على فهم:

مفهوم خطأ القياس (كلاسيكي وغير كلاسيكي).

تأثير خطأ القياس على انحياز التقديرات (Attenuation Bias).

المنهجية الإحصائية لاختبار الاستقلال الشرطي للكشف عن أخطاء القياس.

كيفية تطبيق الاختبار عملياً باستخدام بيانات حقيقية.

🚀 الميزات الرئيسية

محاكاة تفاعلية: لوحة تحكم لتغيير معالم البيانات ومشاهدة تأثير خطأ القياس فوراً على خطوط الانحدار.

شرح نظري مبسط: تبسيط المعادلات الرياضية المعقدة والفرضيات الإحصائية.

تجارب مونت كارلو: عرض نتائج قوة الاختبار (Power) وحجمه (Size) تحت سيناريوهات مختلفة.

تطبيق عملي: عرض نتائج الاختبار على بيانات الأجور الإدارية والمسحية (CPS).

واجهة مستخدم ثنائية اللغة: محتوى معرب بالكامل مع المصطلحات الإنجليزية المقابلة.

🛠 المتطلبات التقنية

لتشغيل هذا التطبيق، ستحتاج إلى تثبيت Python والمكتبات التالية:

streamlit

pandas

numpy

plotly

scipy

💻 طريقة التشغيل

قم بتحميل ملف meas2.py.

افتح نافذة الأوامر (Terminal/CMD).

قم بتثبيت المكتبات اللازمة:

code
Bash
download
content_copy
expand_less
pip install streamlit pandas numpy plotly scipy

قم بتشغيل التطبيق:

code
Bash
download
content_copy
expand_less
streamlit run meas2.py
📚 المراجع العلمية

يعتمد التطبيق بشكل أساسي على:

Wilhelm, D. (2018): "Testing for the Presence of Measurement Error".

Lee, Y.J. & Wilhelm, D. (2019): "Testing for the Presence of Measurement Error in Stata".

Delgado & Gonzalez Manteiga (2001): لمنهجية اختبار الاستقلال الشرطي.

👤 إعداد وتطوير

د. مروان رودان

متخصص في الاقتصاد القياسي والتحليل الإحصائي.

📊 Measurement Error Test App (English Version)

An interactive educational and analytical application designed to explain and simplify the methodology for Testing the Presence of Measurement Error in explanatory variables, based on the works of Wilhelm (2018) and Lee & Wilhelm (2019).

📝 Description

This app helps researchers and econometrics students understand:

Concepts of measurement error (Classical & Non-classical).

The impact of error on estimation bias (Attenuation Bias).

Statistical methodology for testing conditional independence to detect errors.

Practical application using administrative and survey data.

🚀 Key Features

Interactive Simulation: Dashboard to manipulate data parameters and visualize bias in real-time.

Theoretical Breakdown: Simplification of complex formulas and statistical hypotheses.

Monte Carlo Simulations: Visualizing test power and size under various scenarios.

Empirical Application: Results from Administrative vs. Survey earnings data (CPS).

RTL Support: Full Arabic interface with corresponding English terminology.

🛠 Requirements

To run this app, you need Python and the following libraries:

streamlit

pandas

numpy

plotly

scipy

💻 How to Run

Download meas2.py.

Open your Terminal/CMD.

Install dependencies:

code
Bash
download
content_copy
expand_less
pip install streamlit pandas numpy plotly scipy

Run the app:

code
Bash
download
content_copy
expand_less
streamlit run meas2.py
📚 References

Wilhelm, D. (2018): "Testing for the Presence of Measurement Error".

Lee, Y.J. & Wilhelm, D. (2019): "Testing for the Presence of Measurement Error in Stata".

👤 Author

Dr. Marwan Roudane

Econometrics and Statistical Analysis 
numpy
plotly
scipy
