"""
تطبيق تفاعلي شامل لشرح اختبار وجود خطأ القياس
Comprehensive Interactive App for Measurement Error Testing
Based on Wilhelm (2018) and Lee & Wilhelm (2019)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ===== Page Configuration =====
st.set_page_config(
    page_title="اختبار وجود خطأ القياس",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Custom CSS for Arabic RTL and Styling =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap');
    
    .main {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
    }
    
    .stMarkdown {
        direction: rtl;
        text-align: right;
    }
    
    /* جعل الصيغ الرياضية من اليسار لليمين */
    .stLatex, .katex, .katex-display, .MathJax, .MathJax_Display {
        direction: ltr !important;
        text-align: center !important;
    }
    
    /* نقل الشريط الجانبي لليمين */
    [data-testid="stSidebar"] {
        direction: rtl;
        right: 0;
        left: auto !important;
    }
    
    [data-testid="stSidebarContent"] {
        direction: rtl;
    }
    
    .stApp {
        direction: rtl;
    }
    
    section[data-testid="stSidebar"] {
        left: unset !important;
        right: 0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Tajawal', sans-serif !important;
        color: #2e8b57;
    }
    
    .definition-box {
        background: linear-gradient(135deg, #20b2aa 0%, #48d1cc 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(32, 178, 170, 0.3);
        direction: rtl;
    }
    
    .formula-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.3);
        text-align: center;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    
    .example-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 15px;
        color: #555;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(250, 112, 154, 0.3);
    }
    
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 20px;
        border-radius: 15px;
        color: #555;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(168, 237, 234, 0.3);
    }
    
    .key-point {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-right: 5px solid #ff6b6b;
    }
    
    .term-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-right: 4px solid #20b2aa;
    }
    
    .sidebar .sidebar-content {
        direction: rtl;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        direction: rtl;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Tajawal', sans-serif;
        font-size: 16px;
        padding: 10px 20px;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
    }
    
    .step-number {
        background: linear-gradient(135deg, #20b2aa 0%, #48d1cc 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===== Sidebar Navigation =====
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="color: #20b2aa;">📊</h1>
    <h2 style="color: #2e8b57;">اختبار خطأ القياس</h2>
    <p style="color: #888;">Measurement Error Test</p>
    <hr style="margin: 15px 0;">
    <p style="color: #2e8b57; font-weight: bold;">من إعداد</p>
    <p style="color: #20b2aa; font-size: 1.1em; font-weight: bold;">د. مروان رودان</p>
</div>
""", unsafe_allow_html=True)

sections = [
    "🏠 المقدمة والتعريفات",
    "📚 أنواع خطأ القياس",
    "⚠️ تأثير خطأ القياس",
    "🎯 الفرضيات الإحصائية",
    "🔬 منهجية الاختبار",
    "📈 المحاكاة والأمثلة",
    "💻 التطبيق العملي",
    "📖 ملخص ومراجع"
]

selected_section = st.sidebar.radio("اختر القسم:", sections)

# ===== Section 1: Introduction and Definitions =====
if selected_section == "🏠 المقدمة والتعريفات":
    
    st.markdown("""
    <h1 style="text-align: center; color: #2e8b57; font-size: 2.5em;">
        🔍 اختبار وجود خطأ القياس في المتغيرات التفسيرية
    </h1>
    <h3 style="text-align: center; color: #888;">
        Testing for the Presence of Measurement Error
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quote from Griliches
    st.markdown("""
    <div class="definition-box">
        <h3>💬 اقتباس مهم من Griliches (1986)</h3>
        <p style="font-style: italic; font-size: 1.1em;">
        "الاقتصاديون القياسيون لديهم موقف متناقض تجاه البيانات الاقتصادية. على مستوى واحد، 
        'البيانات' هي العالم الذي نريد تفسيره، الحقائق الأساسية التي يدّعي الاقتصاديون توضيحها. 
        وعلى مستوى آخر، هي مصدر كل مشاكلنا. عدم كمالها يجعل عملنا صعباً وأحياناً مستحيلاً."
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📖 ما هو خطأ القياس؟ (Measurement Error)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="term-box">
            <h4>🔤 التعريف بالعربية:</h4>
            <p><strong>خطأ القياس</strong> هو الفرق بين القيمة الحقيقية للمتغير والقيمة المُلاحظة أو المُقاسة.</p>
            <p>بمعنى آخر: عندما نقيس شيئاً ما، القيمة التي نحصل عليها قد تختلف عن القيمة الحقيقية.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="term-box">
            <h4>🔤 Definition in English:</h4>
            <p><strong>Measurement Error</strong> is the difference between the true value of a variable and its observed or measured value.</p>
            <p>In other words: what we measure may differ from the actual truth.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📐 الصيغة الرياضية الأساسية")
    
    st.latex(r"""
    \underbrace{X}_{\text{القيمة الملاحظة}} = \underbrace{X^*}_{\text{القيمة الحقيقية}} + \underbrace{\eta}_{\text{خطأ القياس}}
    """)
    
    st.markdown("""
    <div class="info-box">
        <h4>📌 تفسير المعادلة:</h4>
        <ul>
            <li><strong>X (القيمة الملاحظة - Observed Value):</strong> ما نراه في البيانات</li>
            <li><strong>X* (القيمة الحقيقية - True Value):</strong> القيمة الفعلية التي نريد قياسها</li>
            <li><strong>η (خطأ القياس - Measurement Error):</strong> الفرق بين الاثنين</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎯 لماذا نهتم بخطأ القياس؟")
    
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ المشكلة الرئيسية:</h4>
        <p>خطأ القياس في المتغيرات التفسيرية (المستقلة) يؤدي إلى:</p>
        <ul>
            <li>تقديرات منحازة (Biased Estimates)</li>
            <li>استنتاجات إحصائية خاطئة</li>
            <li>قرارات سياسية مبنية على معلومات غير دقيقة</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📋 مصادر خطأ القياس (حسب Griliches)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="example-box">
            <h4>1️⃣ النموذج الخاطئ (Wrong Model)</h4>
            <p><strong>بالإنجليزية:</strong> Model Misspecification</p>
            <hr>
            <p>معظم النماذج الاقتصادية تهمل بعض الاحتكاكات مثل:</p>
            <ul>
                <li>المنافسة غير الكاملة (Imperfect Competition)</li>
                <li>تكاليف التعديل (Adjustment Costs)</li>
                <li>عدم الانتباه (Inattention)</li>
                <li>سوء تقدير الأسعار (Price Misperceptions)</li>
            </ul>
            <p>هذا يجعل الاختيار الأمثل في نموذج الباحث يختلف عن الاختيار الملاحظ في الواقع.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="example-box">
            <h4>2️⃣ البيانات لا تقيس ما يُفترض (Poor Measurement)</h4>
            <p><strong>بالإنجليزية:</strong> Data Quality Issues</p>
            <hr>
            <p>أمثلة:</p>
            <ul>
                <li>درجات الاختبارات قد لا تقيس المهارات الحقيقية</li>
                <li>الأجور المُبلغ عنها في الاستبيانات تحتوي أخطاء</li>
                <li>البيانات الإدارية قد تكون غير دقيقة</li>
                <li>أخطاء في إدخال البيانات</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 🔬 أمثلة تطبيقية من الأدب الاقتصادي")
    
    examples_data = {
        "المجال": [
            "نظرية الاستثمار (Investment Theory)",
            "تكوين المهارات (Skill Formation)", 
            "عوائد التعليم (Returns to Education)",
            "تأثير النقابات (Union Effects)"
        ],
        "المتغير الحقيقي X*": [
            "Marginal q (q الهامشي)",
            "المهارات الحقيقية",
            "سنوات التعليم الفعلية",
            "حالة العضوية النقابية الفعلية"
        ],
        "المتغير الملاحظ X": [
            "Average q (q المتوسط)",
            "درجات الاختبارات",
            "سنوات التعليم المُبلغ عنها",
            "العضوية المُبلغ عنها"
        ],
        "المصدر": [
            "Hayashi (1982)",
            "Cunha et al. (2010)",
            "Kane & Rouse (1995)",
            "Card (1996)"
        ]
    }
    
    df_examples = pd.DataFrame(examples_data)
    st.dataframe(df_examples, use_container_width=True, hide_index=True)
    
    st.markdown("## 📊 تصور بصري: الفرق بين القيم الحقيقية والملاحظة")
    
    # Interactive visualization
    np.random.seed(42)
    n_points = 100
    x_true = np.random.uniform(0, 10, n_points)
    measurement_error = np.random.normal(0, 1.5, n_points)
    x_observed = x_true + measurement_error
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_true, y=x_observed,
        mode='markers',
        marker=dict(
            size=10,
            color=measurement_error,
            colorscale='RdYlBu',
            showscale=True,
            colorbar=dict(title="خطأ القياس")
        ),
        text=[f"الحقيقي: {t:.2f}<br>الملاحظ: {o:.2f}<br>الخطأ: {e:.2f}" 
              for t, o, e in zip(x_true, x_observed, measurement_error)],
        hoverinfo='text',
        name='الملاحظات'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 10], y=[0, 10],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='خط المساواة (لا خطأ)'
    ))
    
    fig.update_layout(
        title="العلاقة بين القيم الحقيقية والملاحظة",
        xaxis_title="X* (القيمة الحقيقية - True Value)",
        yaxis_title="X (القيمة الملاحظة - Observed Value)",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="success-box">
        <h4>🎓 ملاحظات على الرسم:</h4>
        <ul>
            <li>الخط الأحمر المتقطع يمثل الحالة المثالية حيث لا يوجد خطأ قياس (X = X*)</li>
            <li>النقاط الزرقاء: قياس أقل من الحقيقة (Under-reporting)</li>
            <li>النقاط الحمراء: قياس أكثر من الحقيقة (Over-reporting)</li>
            <li>كلما ابتعدت النقطة عن الخط، زاد خطأ القياس</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== Section 2: Types of Measurement Error =====
elif selected_section == "📚 أنواع خطأ القياس":
    
    st.markdown("""
    <h1 style="text-align: center; color: #2e8b57;">
        📚 أنواع خطأ القياس
    </h1>
    <h3 style="text-align: center; color: #888;">
        Types of Measurement Error
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🔄 التصنيف الرئيسي لخطأ القياس")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="definition-box">
            <h3>1️⃣ خطأ القياس الكلاسيكي</h3>
            <h4>Classical Measurement Error</h4>
            <hr>
            <p><strong>التعريف:</strong> خطأ القياس يكون مستقلاً عن:</p>
            <ul>
                <li>القيمة الحقيقية للمتغير X*</li>
                <li>جميع المتغيرات الأخرى في النموذج</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### الشروط الرياضية:")
        st.latex(r"""
        \begin{aligned}
        E(\eta) &= 0 & \text{(المتوسط صفر)} \\
        Cov(\eta, X^*) &= 0 & \text{(استقلال عن القيمة الحقيقية)} \\
        Cov(\eta, \varepsilon) &= 0 & \text{(استقلال عن خطأ النموذج)}
        \end{aligned}
        """)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h3>2️⃣ خطأ القياس غير الكلاسيكي</h3>
            <h4>Non-Classical Measurement Error</h4>
            <hr>
            <p><strong>التعريف:</strong> خطأ القياس قد يعتمد على:</p>
            <ul>
                <li>القيمة الحقيقية للمتغير X*</li>
                <li>متغيرات أخرى في النموذج</li>
            </ul>
            <p><strong>أمثلة:</strong></p>
            <ul>
                <li>أصحاب الدخل المرتفع يُبلغون بأقل (Under-reporting)</li>
                <li>أصحاب الدخل المنخفض يُبلغون بأكثر (Over-reporting)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### الصيغة العامة:")
        st.latex(r"""
        \eta = \eta(X^*, \text{متغيرات أخرى})
        """)
    
    st.markdown("---")
    
    st.markdown("## 📊 مقارنة بصرية بين النوعين")
    
    np.random.seed(42)
    n = 200
    x_true = np.random.uniform(1, 10, n)
    
    # Classical error
    eta_classical = np.random.normal(0, 1, n)
    x_classical = x_true + eta_classical
    
    # Non-classical error (depends on x_true)
    eta_nonclassical = np.random.normal(0, 0.3 * x_true, n)  # Error increases with x
    x_nonclassical = x_true + eta_nonclassical
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("خطأ القياس الكلاسيكي", "خطأ القياس غير الكلاسيكي"))
    
    # Classical
    fig.add_trace(go.Scatter(
        x=x_true, y=eta_classical,
        mode='markers',
        marker=dict(color='#20b2aa', size=8, opacity=0.6),
        name='كلاسيكي'
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Non-classical
    fig.add_trace(go.Scatter(
        x=x_true, y=eta_nonclassical,
        mode='markers',
        marker=dict(color='#f5576c', size=8, opacity=0.6),
        name='غير كلاسيكي'
    ), row=1, col=2)
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_xaxes(title_text="X* (القيمة الحقيقية)", row=1, col=1)
    fig.update_xaxes(title_text="X* (القيمة الحقيقية)", row=1, col=2)
    fig.update_yaxes(title_text="η (خطأ القياس)", row=1, col=1)
    fig.update_yaxes(title_text="η (خطأ القياس)", row=1, col=2)
    
    fig.update_layout(height=400, template="plotly_white", showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>📌 لاحظ الفرق:</h4>
        <ul>
            <li><strong>الرسم الأيسر (كلاسيكي):</strong> تشتت الخطأ ثابت بغض النظر عن قيمة X* - لا توجد علاقة</li>
            <li><strong>الرسم الأيمن (غير كلاسيكي):</strong> تشتت الخطأ يزداد مع زيادة X* - علاقة واضحة!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📋 أنواع فرعية أخرى لخطأ القياس")
    
    tab1, tab2, tab3 = st.tabs(["التصنيف الخاطئ", "خطأ التقريب", "خطأ عدم الاستجابة"])
    
    with tab1:
        st.markdown("""
        <div class="term-box">
            <h4>🔀 التصنيف الخاطئ (Misclassification Error)</h4>
            <p><strong>بالإنجليزية:</strong> Misclassification</p>
            <hr>
            <p><strong>التعريف:</strong> يحدث عندما يكون المتغير ثنائياً (0 أو 1) ويتم تصنيف بعض الملاحظات بشكل خاطئ.</p>
            <p><strong>مثال:</strong></p>
            <ul>
                <li>شخص موظف (1) يُسجل كعاطل (0) أو العكس</li>
                <li>شخص مريض يُشخص كسليم</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""
        \begin{aligned}
        P(X=1|X^*=0) &= \alpha_{01} & \text{(احتمال التصنيف الخاطئ من 0 إلى 1)} \\
        P(X=0|X^*=1) &= \alpha_{10} & \text{(احتمال التصنيف الخاطئ من 1 إلى 0)}
        \end{aligned}
        """)
    
    with tab2:
        st.markdown("""
        <div class="term-box">
            <h4>🔢 خطأ التقريب (Rounding Error)</h4>
            <p><strong>بالإنجليزية:</strong> Rounding/Heaping Error</p>
            <hr>
            <p><strong>التعريف:</strong> يحدث عندما يميل المستجيبون إلى تقريب إجاباتهم.</p>
            <p><strong>مثال:</strong></p>
            <ul>
                <li>الدخل 48,750 يُبلغ كـ 50,000</li>
                <li>العمر 37 يُبلغ كـ 35 أو 40</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="term-box">
            <h4>❓ خطأ عدم الاستجابة (Non-response Error)</h4>
            <p><strong>بالإنجليزية:</strong> Item Non-response</p>
            <hr>
            <p><strong>التعريف:</strong> يحدث عندما يرفض المستجيبون الإجابة على أسئلة معينة.</p>
            <p><strong>المشكلة:</strong> عدم الاستجابة غالباً ما يكون انتقائياً وليس عشوائياً.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 🔍 خطأ القياس التفاضلي vs غير التفاضلي")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>✅ خطأ القياس غير التفاضلي</h4>
            <h5>Non-Differential Measurement Error</h5>
            <hr>
            <p><strong>التعريف:</strong> خطأ القياس في X لا يعتمد على Y</p>
            <p>رياضياً:</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"Y \perp \!\!\! \perp (X, Z) | X^*")
        st.markdown("بمعنى: X و Z لا يوفران معلومات إضافية عن Y بعد معرفة X*")
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ خطأ القياس التفاضلي</h4>
            <h5>Differential Measurement Error</h5>
            <hr>
            <p><strong>التعريف:</strong> خطأ القياس في X يعتمد على Y</p>
            <p><strong>مثال:</strong> المرضى قد يبلغون عن تعرضهم للعوامل بشكل مختلف عن الأصحاء</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="key-point">
        <h4>💡 نقطة مهمة من ورقة Wilhelm (2018):</h4>
        <p>الاختبار المقترح في الورقة يمكنه اكتشاف مجموعة واسعة من نماذج خطأ القياس غير الكلاسيكي، 
        بما في ذلك العديد من النماذج التي لا يمكن تحديدها (غير قابلة للتعريف - Non-Identified)!</p>
    </div>
    """, unsafe_allow_html=True)

# ===== Section 3: Impact of Measurement Error =====
elif selected_section == "⚠️ تأثير خطأ القياس":
    
    st.markdown("""
    <h1 style="text-align: center; color: #2e8b57;">
        ⚠️ تأثير خطأ القياس على التقديرات
    </h1>
    <h3 style="text-align: center; color: #888;">
        Impact of Measurement Error on Estimates
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 📉 تحيز التخفيف (Attenuation Bias)")
    
    st.markdown("""
    <div class="definition-box">
        <h3>🔤 التعريف:</h3>
        <p><strong>تحيز التخفيف (Attenuation Bias)</strong> هو ميل معاملات الانحدار إلى أن تكون أقرب إلى الصفر 
        (أي أضعف) عندما يوجد خطأ قياس كلاسيكي في المتغير التفسيري.</p>
        <p><strong>بالإنجليزية:</strong> The tendency of regression coefficients to be biased toward zero when there is classical measurement error in the explanatory variable.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📐 الاشتقاق الرياضي")
    
    st.markdown("#### نبدأ بالنموذج الحقيقي:")
    st.latex(r"Y = \alpha + \beta X^* + \varepsilon")
    
    st.markdown("#### لكننا نلاحظ X بدلاً من X*:")
    st.latex(r"X = X^* + \eta")
    
    st.markdown("#### بالتعويض:")
    st.latex(r"Y = \alpha + \beta (X - \eta) + \varepsilon = \alpha + \beta X + (\varepsilon - \beta\eta)")
    
    st.markdown("#### مقدر OLS يعطي:")
    st.latex(r"""
    \hat{\beta}_{OLS} \xrightarrow{p} \frac{Cov(X, Y)}{Var(X)} 
    = \frac{Cov(X^* + \eta, \beta X^* + \varepsilon)}{Var(X^* + \eta)}
    """)
    
    st.markdown("#### تحت فرضيات خطأ القياس الكلاسيكي:")
    st.latex(r"""
    \hat{\beta}_{OLS} \xrightarrow{p} \frac{\beta \cdot Var(X^*)}{Var(X^*) + Var(\eta)} 
    = \beta \cdot \underbrace{\frac{Var(X^*)}{Var(X^*) + Var(\eta)}}_{\text{عامل التخفيف } \lambda}
    """)
    
    st.markdown("""
    <div class="formula-box">
        <h3>📌 النتيجة الرئيسية:</h3>
        <p>عامل التخفيف (Attenuation Factor):</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    \lambda = \frac{\sigma^2_{X^*}}{\sigma^2_{X^*} + \sigma^2_{\eta}} = \frac{\text{التباين الحقيقي}}{\text{التباين الكلي}}
    """)
    
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ ملاحظات مهمة:</h4>
        <ul>
            <li>$0 < \lambda < 1$ دائماً</li>
            <li>كلما زاد تباين خطأ القياس، قل λ وزاد التحيز</li>
            <li>المعامل المقدر يكون أقرب للصفر من المعامل الحقيقي</li>
            <li>هذا يسمى أيضاً <strong>Errors-in-Variables Bias</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎮 محاكاة تفاعلية: شاهد تأثير خطأ القياس")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ إعدادات المحاكاة")
        n_sim = st.slider("حجم العينة (n)", 50, 500, 200, 50)
        true_beta = st.slider("المعامل الحقيقي (β)", 0.5, 3.0, 1.5, 0.1)
        sigma_x = st.slider("انحراف X* (σx)", 0.5, 3.0, 1.5, 0.1)
        sigma_eta = st.slider("انحراف خطأ القياس (ση)", 0.0, 2.0, 0.5, 0.1)
        sigma_eps = st.slider("انحراف خطأ النموذج (σε)", 0.3, 2.0, 0.5, 0.1)
    
    with col2:
        np.random.seed(42)
        x_star = np.random.normal(0, sigma_x, n_sim)
        eta = np.random.normal(0, sigma_eta, n_sim)
        eps = np.random.normal(0, sigma_eps, n_sim)
        
        x_obs = x_star + eta
        y = true_beta * x_star + eps
        
        # True regression
        slope_true = true_beta
        
        # OLS regression (with measurement error)
        if sigma_eta > 0:
            slope_ols = np.cov(x_obs, y)[0,1] / np.var(x_obs)
            lambda_factor = sigma_x**2 / (sigma_x**2 + sigma_eta**2)
        else:
            slope_ols = true_beta
            lambda_factor = 1.0
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=x_obs, y=y,
            mode='markers',
            marker=dict(color='#20b2aa', size=8, opacity=0.5),
            name='البيانات الملاحظة'
        ))
        
        # True line
        x_line = np.linspace(min(x_obs), max(x_obs), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=true_beta * x_line,
            mode='lines',
            line=dict(color='green', width=3),
            name=f'العلاقة الحقيقية (β = {true_beta})'
        ))
        
        # OLS line
        fig.add_trace(go.Scatter(
            x=x_line, y=slope_ols * x_line,
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name=f'خط OLS (β̂ = {slope_ols:.3f})'
        ))
        
        fig.update_layout(
            title="مقارنة العلاقة الحقيقية مع تقدير OLS",
            xaxis_title="X (الملاحظ)",
            yaxis_title="Y",
            height=450,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("المعامل الحقيقي β", f"{true_beta:.3f}")
        with col_m2:
            st.metric("المعامل المقدر β̂", f"{slope_ols:.3f}", 
                     delta=f"{slope_ols - true_beta:.3f}")
        with col_m3:
            st.metric("عامل التخفيف λ", f"{lambda_factor:.3f}")
    
    st.markdown("## 📊 نسبة الإشارة إلى الضوضاء (Signal-to-Noise Ratio)")
    
    st.latex(r"""
    SNR = \frac{Var(X^*)}{Var(\eta)} = \frac{\sigma^2_{X^*}}{\sigma^2_{\eta}}
    """)
    
    st.markdown("""
    <div class="info-box">
        <h4>📌 العلاقة مع عامل التخفيف:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    \lambda = \frac{SNR}{1 + SNR}
    """)
    
    st.markdown("### 📈 تأثير SNR على التحيز")
    
    snr_values = np.linspace(0.1, 10, 100)
    lambda_values = snr_values / (1 + snr_values)
    bias_percent = (1 - lambda_values) * 100
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("عامل التخفيف λ", "نسبة التحيز %"))
    
    fig.add_trace(go.Scatter(
        x=snr_values, y=lambda_values,
        mode='lines',
        line=dict(color='#11998e', width=3),
        name='λ'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=snr_values, y=bias_percent,
        mode='lines',
        line=dict(color='#f5576c', width=3),
        name='التحيز %'
    ), row=1, col=2)
    
    fig.update_xaxes(title_text="SNR", row=1, col=1)
    fig.update_xaxes(title_text="SNR", row=1, col=2)
    fig.update_yaxes(title_text="λ", row=1, col=1)
    fig.update_yaxes(title_text="نسبة التحيز %", row=1, col=2)
    
    fig.update_layout(height=400, template="plotly_white", showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="key-point">
        <h4>💡 الخلاصة:</h4>
        <ul>
            <li>عندما SNR = 1 (تباين الإشارة = تباين الضوضاء): λ = 0.5 أي تحيز 50%!</li>
            <li>للحصول على تحيز أقل من 10%: نحتاج SNR > 9</li>
            <li>لذلك من المهم جداً اختبار وجود خطأ القياس قبل التحليل</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== Section 4: Statistical Hypotheses =====
elif selected_section == "🎯 الفرضيات الإحصائية":
    
    st.markdown("""
    <h1 style="text-align: center; color: #2e8b57;">
        🎯 الفرضيات الإحصائية للاختبار
    </h1>
    <h3 style="text-align: center; color: #888;">
        Statistical Hypotheses for Testing
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🎪 إطار العمل: ماذا نلاحظ وماذا لا نلاحظ؟")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>✅ ما نلاحظه:</h4>
            <ul>
                <li><strong>Y:</strong> متغير النتيجة</li>
                <li><strong>X:</strong> القياس الملاحظ</li>
                <li><strong>Z:</strong> قياس ثانٍ أو أداة</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>❌ ما لا نلاحظه:</h4>
            <ul>
                <li><strong>X*:</strong> القيمة الحقيقية</li>
                <li><strong>η:</strong> خطأ القياس</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>🤔 السؤال:</h4>
            <p>كيف نختبر شيئاً يتعلق بـ X* الذي لا نلاحظه؟</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 📝 الفرضية الأولى: عدم وجود خطأ قياس")
    
    st.markdown("""
    <div class="definition-box">
        <h3>الفرضية الصفرية (Null Hypothesis):</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    H_0^{\text{no ME}}: P(X = X^*) = 1
    """)
    
    st.markdown("""
    <div class="term-box">
        <h4>📖 تفسير الفرضية:</h4>
        <p>نختبر ما إذا كان المتغير الملاحظ X مساوياً للمتغير الحقيقي X* 
        باحتمال 1 (أي دائماً).</p>
        <ul>
            <li>إذا رفضنا H₀: هناك دليل على وجود خطأ قياس</li>
            <li>إذا لم نرفض H₀: لا يوجد دليل كافٍ على وجود خطأ قياس</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📝 الفرضية الثانية: عدم تأثير خطأ القياس على دالة معينة")
    
    st.markdown("""
    <div class="definition-box">
        <h3>فرضية تساوي الدوال (Functional Equality):</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    H_0^{\text{func}}: P\left(E[\Lambda(Y, X) | X^*, X] = 0\right) = 1
    """)
    
    st.markdown("""
    <div class="term-box">
        <h4>📖 تفسير الفرضية:</h4>
        <p>نختبر ما إذا كان خطأ القياس (إن وجد) يؤثر على دالة معينة نهتم بها.</p>
        <p><strong>أمثلة:</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["اختبار تساوي التوقعات الشرطية", "اختبار تساوي معاملات Gini"])
    
    with tab1:
        st.markdown("### تساوي التوقعات الشرطية:")
        st.latex(r"""
        P\left(E[Y|X^*] = E[Y|X]\right) = 1
        """)
        st.markdown("""
        **التفسير:** هل التوقع الشرطي لـ Y معطى X* يساوي التوقع الشرطي لـ Y معطى X؟
        """)
    
    with tab2:
        st.markdown("### تساوي معاملات Gini الشرطية:")
        st.latex(r"""
        P\left(G^*_P(X^*) = G_P(X)\right) = 1
        """)
        st.markdown("""
        **التفسير:** هل مقاييس عدم المساواة متساوية سواء استخدمنا X* أو X؟
        """)
    
    st.markdown("## 🔑 الفرضية الأساسية: قيد الاستبعاد (Exclusion Restriction)")
    
    st.markdown("""
    <div class="formula-box">
        <h3>Assumption 1: قيد الاستبعاد</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    Y \perp \!\!\! \perp Z \mid X^*
    """)
    
    st.markdown("""
    <div class="info-box">
        <h4>📖 تفسير الفرضية:</h4>
        <p><strong>Y مستقل عن Z شرطياً على X*</strong></p>
        <p>بمعنى آخر: Z يؤثر على Y فقط من خلال X*، وليس مباشرة.</p>
        <p><strong>مثال:</strong> إذا كان X* هو التعليم الحقيقي و Z هو المسافة من الجامعة:</p>
        <ul>
            <li>المسافة تؤثر على الأجور (Y) فقط من خلال تأثيرها على التعليم (X*)</li>
            <li>بمجرد معرفة التعليم الحقيقي، المسافة لا تعطي معلومات إضافية عن الأجور</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 💡 الفكرة الأساسية: تحويل الفرضية إلى شيء قابل للاختبار")
    
    st.markdown("""
    <div class="key-point">
        <h4>🎯 النظرية الأساسية (Theorem 1):</h4>
        <p>تحت قيد الاستبعاد، الفرضية الصفرية لعدم وجود خطأ قياس:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    H_0: P(X = X^*) = 1
    """)
    
    st.markdown("### تستلزم (تُضمن) الشرط الملاحظ:")
    
    st.latex(r"""
    Y \perp \!\!\! \perp Z \mid X
    """)
    
    st.markdown("""
    <div class="success-box">
        <h4>✨ لماذا هذا مهم؟</h4>
        <ul>
            <li>الشرط الثاني يعتمد فقط على المتغيرات الملاحظة (Y, X, Z)</li>
            <li>يمكننا اختباره مباشرة بدون الحاجة لمعرفة X*</li>
            <li>إذا رفضنا الشرط الثاني ← نرفض عدم وجود خطأ قياس</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📊 رسم توضيحي للعلاقات")
    
    st.markdown("""
    ```
    الحالة بدون خطأ قياس:                    الحالة مع خطأ قياس:
    
         Z                                        Z
         │                                        │
         ▼                                        ▼
        X* ────────► Y                           X* ────────► Y
         │                                        │
         │ (X = X*)                                │ (X ≠ X*)
         ▼                                        ▼
         X                                        X
         
    Y ⊥ Z | X ✓                               Y ⊥ Z | X ✗
    ```
    """)
    
    st.markdown("## 🔄 التكافؤ: متى يكون الاختبار له قوة؟")
    
    st.markdown("""
    <div class="definition-box">
        <h3>النظرية 2: شروط التكافؤ</h3>
        <p>تحت الشروط التالية، الفرضية الصفرية مكافئة للشرط الملاحظ:</p>
    </div>
    """, unsafe_allow_html=True)
    
    conditions = {
        "الشرط": [
            "قيد الاستبعاد الأقوى",
            "الرتابة (Monotonicity)",
            "الهيمنة العشوائية من الدرجة الأولى (FOSD)"
        ],
        "الصيغة الرياضية": [
            r"$Y \perp (X, Z) | X^*$",
            r"$E[\mu(Y)|X^*=x^*]$ رتيبة في $x^*$",
            r"$P(X^* \geq x^*|X, Z=z_1) \leq P(X^* \geq x^*|X, Z=z_2)$"
        ],
        "التفسير": [
            "X و Z لا يؤثران على Y إلا من خلال X*",
            "العلاقة بين Y و X* رتيبة (مثل دالة إنتاج)",
            "Z له علاقة كافية مع X* (شرط الصلة)"
        ]
    }
    
    df_conditions = pd.DataFrame(conditions)
    st.dataframe(df_conditions, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="key-point">
        <h4>💡 أهمية التكافؤ:</h4>
        <ul>
            <li><strong>الاتجاه الأول:</strong> H₀ → Y ⊥ Z | X (صلاحية الاختبار)</li>
            <li><strong>الاتجاه الثاني:</strong> Y ⊥ Z | X → H₀ (قوة الاختبار)</li>
            <li>بدون التكافؤ، قد يفشل الاختبار في اكتشاف خطأ القياس حتى لو كان موجوداً</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== Section 5: Test Methodology =====
elif selected_section == "🔬 منهجية الاختبار":
    
    st.markdown("""
    <h1 style="text-align: center; color: #2e8b57;">
        🔬 منهجية الاختبار
    </h1>
    <h3 style="text-align: center; color: #888;">
        Test Methodology (Delgado & Gonzalez Manteiga, 2001)
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 📋 الخطوات الأساسية للاختبار")
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 الهدف:</h4>
        <p>اختبار الاستقلال الشرطي:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    H_0: E[Y|X, Z] = E[Y|X] \quad \text{(تقريباً في كل مكان)}
    """)
    
    st.markdown("### الخطوة 1️⃣: إعادة صياغة الفرضية")
    
    st.markdown("""
    <div class="term-box">
        <p>نعيد صياغة الفرضية الصفرية كالتالي:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    H_0: E[T(X, Z)] = 0
    """)
    
    st.markdown("حيث:")
    
    st.latex(r"""
    T(x, z) = E\left[f_X(X)\{Y - E[Y|X]\}\mathbf{1}\{X \leq x\}\mathbf{1}\{Z \leq z\}\right]
    """)
    
    st.markdown("""
    <div class="term-box">
        <h4>📖 شرح المكونات:</h4>
        <ul>
            <li><strong>f_X(X):</strong> دالة الكثافة للمتغير X</li>
            <li><strong>Y - E[Y|X]:</strong> البواقي (residuals) من انحدار Y على X</li>
            <li><strong>1{X ≤ x}:</strong> دالة مؤشر (indicator function)</li>
            <li><strong>1{Z ≤ z}:</strong> دالة مؤشر أخرى</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### الخطوة 2️⃣: حساب إحصائية الاختبار")
    
    st.markdown("""
    <div class="formula-box">
        <h4>النظير التجريبي (Empirical Analogue):</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    T_n(x, z) = \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} \frac{1}{h} K\left(\frac{X_i - X_j}{h}\right)
    (Y_i - Y_j) \mathbf{1}\{X_i \leq x\}\mathbf{1}\{Z_i \leq z\}
    """)
    
    st.markdown("""
    <div class="term-box">
        <h4>📖 شرح المكونات:</h4>
        <ul>
            <li><strong>n:</strong> حجم العينة</li>
            <li><strong>h:</strong> معلمة النطاق (bandwidth)</li>
            <li><strong>K(·):</strong> دالة النواة (kernel function)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### الخطوة 3️⃣: اختيار نوع إحصائية الاختبار")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>📊 إحصائية Cramér-von Mises (CvM)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""
        T_n^{CvM} = n \sum_{i=1}^{n} T_n(X_i, Z_i)^2
        """)
        
        st.markdown("""
        <p><strong>المميزات:</strong></p>
        <ul>
            <li>تجمع المعلومات من كل نقاط البيانات</li>
            <li>أكثر استقراراً</li>
        </ul>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>📊 إحصائية Kolmogorov-Smirnov (KS)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""
        T_n^{KS} = \sup_{x,z} |\sqrt{n} T_n(x, z)|
        """)
        
        st.markdown("""
        <p><strong>المميزات:</strong></p>
        <ul>
            <li>تركز على أقصى انحراف</li>
            <li>حساسة للانحرافات المحلية</li>
        </ul>
        """, unsafe_allow_html=True)
    
    st.markdown("### الخطوة 4️⃣: حساب القيم الحرجة باستخدام Bootstrap")
    
    st.markdown("""
    <div class="definition-box">
        <h3>طريقة Multiplier Bootstrap:</h3>
        <p>نولد عينات Bootstrap باستخدام متغير مضاعف V له:</p>
        <ul>
            <li>E[V] = 0</li>
            <li>Var[V] = 1</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### توزيع Mammen (الافتراضي):")
    
    st.latex(r"""
    V = \begin{cases}
    \frac{-(\sqrt{5}-1)}{2} & \text{باحتمال } p = \frac{\sqrt{5}+1}{2\sqrt{5}} \\[10pt]
    \frac{\sqrt{5}+1}{2} & \text{باحتمال } 1-p
    \end{cases}
    """)
    
    st.markdown("### الخطوة 5️⃣: اتخاذ القرار")
    
    st.markdown("""
    <div class="key-point">
        <h4>🎯 قاعدة القرار:</h4>
        <ul>
            <li>إذا كانت إحصائية الاختبار > القيمة الحرجة ← نرفض H₀</li>
            <li>إذا كانت إحصائية الاختبار ≤ القيمة الحرجة ← لا نرفض H₀</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔧 اختيار دالة النواة (Kernel Function)")
    
    kernels_data = {
        "النواة": ["Epanechnikov", "Gaussian", "Uniform", "Triangular", "Biweight"],
        "الصيغة": [
            r"$\frac{3}{4}(1-u^2)\mathbf{1}_{|u|\leq 1}$",
            r"$\frac{1}{\sqrt{2\pi}}e^{-u^2/2}$",
            r"$\frac{1}{2}\mathbf{1}_{|u|\leq 1}$",
            r"$(1-|u|)\mathbf{1}_{|u|\leq 1}$",
            r"$\frac{15}{16}(1-u^2)^2\mathbf{1}_{|u|\leq 1}$"
        ],
        "الخصائص": [
            "الأمثل نظرياً (Optimal)",
            "سلس، دعم غير محدود",
            "بسيط، غير سلس",
            "سلس، دعم محدود",
            "أكثر سلاسة من Epanechnikov"
        ]
    }
    
    df_kernels = pd.DataFrame(kernels_data)
    st.dataframe(df_kernels, use_container_width=True, hide_index=True)
    
    # Visualize kernels
    st.markdown("### 📈 تصور دوال النواة")
    
    u = np.linspace(-2, 2, 200)
    
    fig = go.Figure()
    
    # Epanechnikov
    k_epan = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
    fig.add_trace(go.Scatter(x=u, y=k_epan, name='Epanechnikov', 
                             line=dict(width=2)))
    
    # Gaussian
    k_gauss = (1/np.sqrt(2*np.pi)) * np.exp(-u**2/2)
    fig.add_trace(go.Scatter(x=u, y=k_gauss, name='Gaussian', 
                             line=dict(width=2)))
    
    # Uniform
    k_uniform = np.where(np.abs(u) <= 1, 0.5, 0)
    fig.add_trace(go.Scatter(x=u, y=k_uniform, name='Uniform', 
                             line=dict(width=2)))
    
    # Triangular
    k_tri = np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)
    fig.add_trace(go.Scatter(x=u, y=k_tri, name='Triangular', 
                             line=dict(width=2)))
    
    fig.update_layout(
        title="مقارنة دوال النواة المختلفة",
        xaxis_title="u",
        yaxis_title="K(u)",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## 📏 اختيار معلمة النطاق (Bandwidth)")
    
    st.markdown("""
    <div class="info-box">
        <h4>القاعدة الافتراضية (Rule of Thumb):</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    h = n^{-1/(3q)}
    """)
    
    st.markdown("""
    حيث:
    - n: حجم العينة
    - q: بُعد المتغير (X, W₁)
    """)
    
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ تحذير:</h4>
        <p>اختيار h مهم جداً:</p>
        <ul>
            <li>h صغير جداً → تباين عالي</li>
            <li>h كبير جداً → تحيز عالي</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== Section 6: Simulations and Examples =====
elif selected_section == "📈 المحاكاة والأمثلة":
    
    st.markdown("""
    <h1 style="text-align: center; color: #2e8b57;">
        📈 المحاكاة والأمثلة التطبيقية
    </h1>
    <h3 style="text-align: center; color: #888;">
        Monte Carlo Simulations and Examples
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🎮 محاكاة تفاعلية")
    
    st.markdown("""
    <div class="info-box">
        <h4>📋 نموذج المحاكاة:</h4>
        <p>معادلة النتيجة:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    Y = (X^*)^2 + \frac{1}{2}X^* + \varepsilon, \quad \varepsilon \sim N(0, \sigma_\varepsilon^2)
    """)
    
    st.markdown("### النماذج المختلفة لنظام القياس:")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "النموذج I: كلاسيكي", 
        "النموذج II: غير متجانس",
        "النموذج III: تابع مزدوج",
        "النموذج IV: علاقة غير خطية"
    ])
    
    with tab1:
        st.latex(r"""
        \begin{aligned}
        X &= X^* + D \cdot N(0, \sigma_{ME}^2) \\
        Z &= X^* + N(0, 0.3^2)
        \end{aligned}
        """)
        st.markdown("خطأ قياس كلاسيكي مستقل عن X*")
    
    with tab2:
        st.latex(r"""
        \begin{aligned}
        X &= X^* + D \cdot N(0, \sigma_{ME}^2) \cdot e^{-|X^*-0.5|} \\
        Z &= X^* + N(0, 0.3^2)
        \end{aligned}
        """)
        st.markdown("تباين الخطأ يعتمد على X* (Heteroskedastic)")
    
    with tab3:
        st.latex(r"""
        \begin{aligned}
        X &= X^* + D \cdot N(0, \sigma_{ME}^2) \cdot e^{-|X^*-0.5|} \\
        Z &= X^* + N(0, 0.3^2) \cdot e^{-|X^*-0.5|}
        \end{aligned}
        """)
        st.markdown("كلا الخطأين يعتمدان على X*")
    
    with tab4:
        st.latex(r"""
        \begin{aligned}
        X &= X^* + D \cdot N(0, \sigma_{ME}^2) \\
        Z &= -(X^*-1)^2 + N(0, 0.2^2)
        \end{aligned}
        """)
        st.markdown("العلاقة بين Z و X* غير خطية")
    
    st.markdown("---")
    
    st.markdown("## 🎯 محاكاة حية")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ إعدادات:")
        n_sim = st.selectbox("حجم العينة", [200, 500], index=0)
        model_type = st.selectbox("نوع النموذج", 
                                  ["I: كلاسيكي", "II: غير متجانس", 
                                   "III: تابع مزدوج", "IV: غير خطي"])
        sigma_me = st.slider("σ_ME", 0.0, 1.0, 0.5, 0.1)
        prob_me = st.slider("احتمال خطأ القياس (1-λ)", 0.0, 1.0, 0.5, 0.1)
        
        run_sim = st.button("🚀 تشغيل المحاكاة", type="primary")
    
    with col2:
        if run_sim or 'sim_results' not in st.session_state:
            np.random.seed(42)
            
            x_star = np.random.uniform(0, 1, n_sim)
            D = np.random.binomial(1, prob_me, n_sim)
            sigma_eps = 0.5 if model_type != "IV: غير خطي" else 0.2
            eps = np.random.normal(0, sigma_eps, n_sim)
            
            if model_type == "I: كلاسيكي":
                eta_x = np.random.normal(0, sigma_me, n_sim)
                eta_z = np.random.normal(0, 0.3, n_sim)
                X = x_star + D * eta_x
                Z = x_star + eta_z
                
            elif model_type == "II: غير متجانس":
                scale_factor = np.exp(-np.abs(x_star - 0.5))
                eta_x = np.random.normal(0, sigma_me, n_sim) * scale_factor
                eta_z = np.random.normal(0, 0.3, n_sim)
                X = x_star + D * eta_x
                Z = x_star + eta_z
                
            elif model_type == "III: تابع مزدوج":
                scale_factor = np.exp(-np.abs(x_star - 0.5))
                eta_x = np.random.normal(0, sigma_me, n_sim) * scale_factor
                eta_z = np.random.normal(0, 0.3, n_sim) * scale_factor
                X = x_star + D * eta_x
                Z = x_star + eta_z
                
            else:  # IV: غير خطي
                eta_x = np.random.normal(0, sigma_me, n_sim)
                eta_z = np.random.normal(0, 0.2, n_sim)
                X = x_star + D * eta_x
                Z = -(x_star - 1)**2 + eta_z
            
            Y = x_star**2 + 0.5 * x_star + eps
            
            st.session_state['sim_data'] = {'X': X, 'Y': Y, 'Z': Z, 'X_star': x_star}
        
        if 'sim_data' in st.session_state:
            data = st.session_state['sim_data']
            
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=("Y vs X", "X vs Z"))
            
            fig.add_trace(go.Scatter(
                x=data['X'], y=data['Y'],
                mode='markers',
                marker=dict(color='#20b2aa', size=6, opacity=0.6),
                name='Y vs X'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data['Z'], y=data['X'],
                mode='markers',
                marker=dict(color='#11998e', size=6, opacity=0.6),
                name='X vs Z'
            ), row=1, col=2)
            
            fig.update_xaxes(title_text="X", row=1, col=1)
            fig.update_yaxes(title_text="Y", row=1, col=1)
            fig.update_xaxes(title_text="Z", row=1, col=2)
            fig.update_yaxes(title_text="X", row=1, col=2)
            
            fig.update_layout(height=400, template="plotly_white", showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## 📊 نتائج المحاكاة من الورقة")
    
    st.markdown("""
    <div class="info-box">
        <h4>📋 إعدادات المحاكاة الأصلية:</h4>
        <ul>
            <li>1000 تكرار لكل إعداد</li>
            <li>Bootstrap: 100 عينة</li>
            <li>مستوى الدلالة: 5%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulation results table
    results_data = {
        "n": [200, 200, 200, 500, 500, 500],
        "النموذج": ["I", "II", "III", "I", "II", "III"],
        "σ_ME=0.2": [0.164, 0.123, 0.149, 0.270, 0.190, 0.235],
        "σ_ME=0.5": [0.394, 0.322, 0.399, 0.777, 0.630, 0.782],
        "σ_ME=1.0": [0.319, 0.370, 0.472, 0.683, 0.755, 0.875]
    }
    
    df_results = pd.DataFrame(results_data)
    
    st.markdown("### احتمالات الرفض (1-λ = 0.25):")
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    st.markdown("### 📈 رسم بياني للقوة")
    
    # Power curve visualization
    lambda_values = [0, 0.25, 0.5, 0.75, 1.0]
    power_model1 = [0.049, 0.394, 0.853, 0.981, 0.995]
    power_model2 = [0.049, 0.322, 0.767, 0.956, 0.992]
    power_model3 = [0.051, 0.399, 0.876, 0.986, 1.000]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lambda_values, y=power_model1,
        mode='lines+markers',
        name='النموذج I',
        line=dict(color='#20b2aa', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=lambda_values, y=power_model2,
        mode='lines+markers',
        name='النموذج II',
        line=dict(color='#11998e', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=lambda_values, y=power_model3,
        mode='lines+markers',
        name='النموذج III',
        line=dict(color='#f5576c', width=3)
    ))
    
    fig.add_hline(y=0.05, line_dash="dash", line_color="gray",
                  annotation_text="مستوى الدلالة 5%")
    
    fig.update_layout(
        title="منحنيات القوة للاختبار (n=200, σ_ME=0.5)",
        xaxis_title="1-λ (احتمال خطأ القياس)",
        yaxis_title="احتمال الرفض",
        height=450,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="success-box">
        <h4>✅ الملاحظات الرئيسية:</h4>
        <ul>
            <li>الاختبار يتحكم في الحجم جيداً تحت H₀ (عند 1-λ = 0)</li>
            <li>القوة تزداد مع زيادة احتمال خطأ القياس</li>
            <li>القوة تزداد مع زيادة حجم العينة</li>
            <li>الاختبار له قوة ضد جميع أنواع خطأ القياس</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== Section 7: Practical Application =====
elif selected_section == "💻 التطبيق العملي":
    
    st.markdown("""
    <h1 style="text-align: center; color: #2e8b57;">
        💻 التطبيق العملي: البيانات الإدارية
    </h1>
    <h3 style="text-align: center; color: #888;">
        Empirical Application: Administrative Earnings Data
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 📋 البيانات المستخدمة")
    
    st.markdown("""
    <div class="info-box">
        <h4>مصدر البيانات:</h4>
        <p><strong>1978 Current Population Survey - Social Security Earnings Records Exact Match File</strong></p>
        <ul>
            <li>بيانات مسحية من CPS 1978</li>
            <li>سجلات الأجور الإدارية من الضمان الاجتماعي</li>
            <li>إمكانية المطابقة بين المصدرين</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="definition-box">
            <h4>Y: الأجور المسحية</h4>
            <p>(Survey Earnings 1977)</p>
            <p>من استبيان CPS</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>X: الأجور الإدارية</h4>
            <p>(Admin Earnings 1977)</p>
            <p>من سجلات الضمان الاجتماعي</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-box">
            <h4>Z: الأجور الإدارية السابقة</h4>
            <p>(Admin Earnings 1976)</p>
            <p>من سجلات السنة السابقة</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 🔬 النموذج الاقتصادي")
    
    st.latex(r"""
    E_2^* = h(E_1^*, U)
    """)
    
    st.markdown("""
    <div class="term-box">
        <h4>📖 التفسير:</h4>
        <ul>
            <li>$E_2^*$: الأجور الحقيقية في الفترة 2</li>
            <li>$E_1^*$: الأجور الحقيقية في الفترة 1</li>
            <li>$U$: صدمات للأجور</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### نظام القياس:")
    
    st.latex(r"""
    \begin{aligned}
    A_t &= m_{A_t}(E_t^*, \eta_{A_t}), \quad t = 1, 2 \\
    S_2 &= m_{S_2}(E_2^*, \eta_{S_2})
    \end{aligned}
    """)
    
    st.markdown("## ✅ تحقق قيد الاستبعاد")
    
    st.markdown("""
    <div class="key-point">
        <h4>💡 لماذا يصح قيد الاستبعاد؟</h4>
        <p>الشرط المطلوب:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    \eta_{S_2} \perp \!\!\! \perp (\eta_{A_1}, U) \mid E_2^*
    """)
    
    st.markdown("""
    <div class="info-box">
        <h4>📖 التبرير:</h4>
        <ul>
            <li>خطأ القياس في المسح (ηS2) له مصادر مختلفة تماماً عن خطأ القياس الإداري (ηA1)</li>
            <li>المسح يُجمع بواسطة محققين، في المنزل، في وقت مختلف</li>
            <li>السجلات الإدارية تُجمع من أصحاب العمل</li>
            <li>لا يوجد سبب لاعتقاد وجود علاقة مباشرة بين الخطأين</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📊 نتائج الاختبار")
    
    results_empirical = {
        "العينة": [
            "العينة الكاملة",
            "الأجور في IQR",
            "الذكور البيض",
            "+ أعزب",
            "+ عمر [25,65]",
            "+ دوام كامل (*)",
            "+ ثانوية فأكثر",
            "+ أجور في IQR"
        ],
        "إحصائية الاختبار": [0.151, 0.401, 0.073, 0.216, 0.009, 0.010, 0.009, 0.053],
        "p-value": [0.000, 0.000, 0.000, 0.000, 0.012, 0.017, 0.030, 0.012],
        "القيمة الحرجة 5%": [0.007, 0.024, 0.004, 0.015, 0.007, 0.009, 0.008, 0.037],
        "حجم العينة": [31228, 15614, 12591, 5043, 1669, 972, 867, 342]
    }
    
    df_empirical = pd.DataFrame(results_empirical)
    
    st.dataframe(df_empirical, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ الاستنتاج الرئيسي:</h4>
        <p>نرفض الفرضية الصفرية لعدم وجود خطأ قياس في جميع العينات!</p>
        <ul>
            <li>p-values قريبة من الصفر أو صغيرة جداً</li>
            <li>هناك دليل قوي على وجود خطأ قياس في البيانات الإدارية</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📈 تصور البيانات")
    
    # Simulated data similar to empirical
    np.random.seed(42)
    n = 1000
    admin_77 = np.random.lognormal(8.5, 0.7, n)
    admin_77 = np.clip(admin_77, 0, 16500)
    admin_76 = admin_77 * np.random.uniform(0.8, 1.2, n)
    survey_77 = admin_77 + np.random.normal(0, 1000, n)
    
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=("توزيع الأجور", "الفرق بين المصدرين"))
    
    fig.add_trace(go.Histogram(
        x=admin_77, name='إداري',
        marker_color='#20b2aa', opacity=0.7
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=survey_77, name='مسحي',
        marker_color='#11998e', opacity=0.7
    ), row=1, col=1)
    
    diff = admin_77 - survey_77
    fig.add_trace(go.Histogram(
        x=diff, name='الفرق',
        marker_color='#f5576c', opacity=0.7
    ), row=1, col=2)
    
    fig.update_layout(height=400, template="plotly_white", barmode='overlay')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## 💻 كود Stata للاختبار")
    
    st.code("""
. dgmtest repearn77 ssearn77 ssearn76, bootnum(5000)

-----------------------------------------------------
 Delgado and Manteiga test
-----------------------------------------------------
H0: E[Y | X,W1,Z] = E[Y | X,W1]

----- parameter settings -----
Test statistic: CvM (default)
Kernel: epanechnikov (default)
bw = n^(1/3q) (default)
bootstrap multiplier distribution: mammen (default)

number of observations: 2682
bandwidth: .07197479

----- test results -----
CvM = .51238949
bootstrap critical value at 1%: .63053938
bootstrap critical value at 5%: .41803533
bootstrap critical value at 10%: .33279162
p(CvM < CvM*) = .0262
    """, language="stata")

# ===== Section 8: Summary and References =====
elif selected_section == "📖 ملخص ومراجع":
    
    st.markdown("""
    <h1 style="text-align: center; color: #2e8b57;">
        📖 ملخص ومراجع
    </h1>
    <h3 style="text-align: center; color: #888;">
        Summary and References
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 📝 ملخص النقاط الرئيسية")
    
    st.markdown("""
    <div class="definition-box">
        <h3>1️⃣ ما هو خطأ القياس؟</h3>
        <p>الفرق بين القيمة الحقيقية (X*) والقيمة الملاحظة (X) للمتغير.</p>
        <p><strong>X = X* + η</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <h3>2️⃣ لماذا هو مهم؟</h3>
        <ul>
            <li>يسبب تحيز التخفيف (Attenuation Bias)</li>
            <li>يؤدي إلى استنتاجات خاطئة</li>
            <li>يؤثر على القرارات السياسية</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <h3>3️⃣ كيف نختبره؟</h3>
        <p>بدلاً من اختبار H₀: P(X = X*) = 1 مباشرة:</p>
        <ol>
            <li>نستخدم قيد الاستبعاد: Y ⊥ Z | X*</li>
            <li>نحوله إلى شرط قابل للاختبار: Y ⊥ Z | X</li>
            <li>نستخدم اختبار Delgado & Gonzalez Manteiga</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>4️⃣ مميزات هذا النهج:</h3>
        <ul>
            <li>لا يتطلب تحديد (identification) النموذج</li>
            <li>يعمل مع خطأ القياس غير الكلاسيكي</li>
            <li>لا يحتاج افتراضات parametric</li>
            <li>سهل التنفيذ</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📚 المصطلحات الرئيسية")
    
    terms = {
        "المصطلح بالعربية": [
            "خطأ القياس",
            "خطأ القياس الكلاسيكي",
            "خطأ القياس غير الكلاسيكي",
            "تحيز التخفيف",
            "قيد الاستبعاد",
            "الاستقلال الشرطي",
            "دالة النواة",
            "معلمة النطاق",
            "التصنيف الخاطئ",
            "نسبة الإشارة إلى الضوضاء"
        ],
        "المصطلح بالإنجليزية": [
            "Measurement Error",
            "Classical Measurement Error",
            "Non-Classical Measurement Error",
            "Attenuation Bias",
            "Exclusion Restriction",
            "Conditional Independence",
            "Kernel Function",
            "Bandwidth",
            "Misclassification",
            "Signal-to-Noise Ratio (SNR)"
        ],
        "الرمز": [
            "η",
            "Cov(η, X*) = 0",
            "Cov(η, X*) ≠ 0",
            "λ = σ²ₓ*/(σ²ₓ* + σ²η)",
            "Y ⊥ Z | X*",
            "Y ⊥ Z | X",
            "K(·)",
            "h",
            "α₀₁, α₁₀",
            "σ²ₓ*/σ²η"
        ]
    }
    
    df_terms = pd.DataFrame(terms)
    st.dataframe(df_terms, use_container_width=True, hide_index=True)
    
    st.markdown("## 📖 المراجع الرئيسية")
    
    st.markdown("""
    ### الأوراق الأساسية:
    
    1. **Wilhelm, D. (2018)**. "Testing for the Presence of Measurement Error." 
       *CeMMAP Working Paper CWP45/18*.
       
    2. **Lee, Y.J. & Wilhelm, D. (2019)**. "Testing for the Presence of Measurement Error in Stata."
       *CeMMAP Working Paper CWP47/19*.
       
    3. **Delgado, M.A. & Gonzalez Manteiga, W. (2001)**. "Significance Testing in Nonparametric 
       Regression Based on the Bootstrap." *The Annals of Statistics*, 29(5), 1469-1507.
    
    ### مراجع إضافية:
    
    4. **Griliches, Z. (1986)**. "Economic Data Issues." *Handbook of Econometrics*, Vol. III.
    
    5. **Bound, J., Brown, C., & Mathiowetz, N. (2001)**. "Measurement Error in Survey Data."
       *Handbook of Econometrics*, Vol. V.
       
    6. **Hausman, J.A. (1978)**. "Specification Tests in Econometrics." 
       *Econometrica*, 46(6), 1251-1271.
       
    7. **Cunha, F., Heckman, J.J., & Schennach, S.M. (2010)**. "Estimating the Technology of 
       Cognitive and Noncognitive Skill Formation." *Econometrica*, 78(3), 883-931.
    """)
    
    st.markdown("## 🔗 روابط مفيدة")
    
    st.markdown("""
    <div class="key-point">
        <h4>💻 الكود المصدري:</h4>
        <ul>
            <li><strong>R:</strong> github.com/danielwilhelm/R-ME-test</li>
            <li><strong>Stata:</strong> github.com/danielwilhelm/STATA-ME-test</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎓 نصائح للتطبيق العملي")
    
    st.markdown("""
    <div class="success-box">
        <h4>✅ قبل إجراء الاختبار:</h4>
        <ol>
            <li>تأكد من صحة قيد الاستبعاد في سياقك</li>
            <li>تحقق من شرط الصلة (relevance) بين Z و X*</li>
            <li>فكر في شرط الرتابة (monotonicity)</li>
        </ol>
    </div>
    
    <div class="warning-box">
        <h4>⚠️ تحذيرات:</h4>
        <ul>
            <li>عدم رفض H₀ لا يعني بالضرورة عدم وجود خطأ قياس</li>
            <li>قد يكون الخطأ صغيراً جداً مقارنة بضوضاء العينة</li>
            <li>اختيار bandwidth مهم جداً</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #20b2aa 0%, #48d1cc 100%); border-radius: 15px; color: white;">
        <h2>🎉 شكراً لاستخدام هذا التطبيق!</h2>
        <p>نأمل أن يكون هذا الشرح مفيداً لفهم اختبار وجود خطأ القياس</p>
        <p style="font-size: 0.9em;">تم إعداده بناءً على أوراق Wilhelm (2018) و Lee & Wilhelm (2019)</p>
        <hr style="border-color: rgba(255,255,255,0.3); margin: 20px 0;">
        <p style="font-size: 1.2em; font-weight: bold;">من إعداد: د. مروان رودان</p>
    </div>
    """, unsafe_allow_html=True)

# ===== Footer =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>📊 تطبيق تفاعلي لشرح اختبار وجود خطأ القياس</p>
    <p>Based on Wilhelm (2018) and Lee & Wilhelm (2019)</p>
    <p style="color: #20b2aa; font-weight: bold; margin-top: 10px;">من إعداد: د. مروان رودان</p>
</div>
""", unsafe_allow_html=True)
