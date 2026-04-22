"""
MIT807: Artificial Intelligence & Its Business Applications
Group 1 — Telecom Customer Churn Prediction
Desktop Application (CustomTkinter)
"""

import os
import sys
import threading
import joblib
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "rf_pipeline.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}\n\n"
            "Please run  train_model.py  from the cus-churn directory first."
        )
    return joblib.load(MODEL_PATH)


def make_label(parent, text, small=False, muted=False):
    size   = 12 if small else 13
    weight = "normal" if (small or muted) else "bold"
    color  = "#94a3b8" if muted else "#e2e8f0"
    return ctk.CTkLabel(parent, text=text, font=("Inter", size, weight),
                        text_color=color, anchor="w")


def make_dropdown(parent, values, width=180):
    return ctk.CTkOptionMenu(
        parent, values=values, width=width,
        fg_color="#1e1b4b", button_color="#4c1d95",
        button_hover_color="#6d28d9",
        text_color="#f1f5f9", font=("Inter", 13),
    )


def make_entry(parent, placeholder="", width=180):
    return ctk.CTkEntry(
        parent, placeholder_text=placeholder, width=width,
        fg_color="#1e1b4b", border_color="#4c1d95",
        text_color="#f1f5f9", placeholder_text_color="#64748b",
        font=("Inter", 13),
    )


def section_frame(parent, title):
    """Returns a labelled card frame."""
    outer = ctk.CTkFrame(parent, fg_color="#1e1b4b",
                         corner_radius=12, border_width=1,
                         border_color="#312e81")
    title_lbl = ctk.CTkLabel(
        outer, text=title,
        font=("Inter", 12, "bold"),
        text_color="#a78bfa",
        anchor="w",
    )
    title_lbl.pack(anchor="w", padx=14, pady=(10, 0))
    sep = ctk.CTkFrame(outer, height=1, fg_color="#312e81")
    sep.pack(fill="x", padx=14, pady=(4, 8))
    inner = ctk.CTkFrame(outer, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=14, pady=(0, 12))
    return outer, inner


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class ChurnApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Churn Predictor  ·  MIT807  ·  GROUP 1")
        self.geometry("1100x820")
        self.minsize(900, 700)
        self.configure(fg_color="#0f0c29")

        # load model in background so UI renders first
        self._model = None
        self._model_error = None
        threading.Thread(target=self._bg_load_model, daemon=True).start()

        self._build_ui()

    # ── Model loading ──────────────────────────────────────────────────────
    def _bg_load_model(self):
        try:
            self._model = load_model()
        except Exception as exc:
            self._model_error = str(exc)

    # ── UI Construction ────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Hero banner ──────────────────────────────────────────────────
        hero = ctk.CTkFrame(self, fg_color="#4c1d95", corner_radius=16)
        hero.pack(fill="x", padx=20, pady=(16, 6))

        badge = ctk.CTkLabel(hero,
                             text="📡  MIT807 · Artificial Intelligence & Its Business Applications",
                             font=("Inter", 11), text_color="rgba(255,255,255,0.75)")
        badge.pack(anchor="w", padx=20, pady=(12, 0))

        ctk.CTkLabel(hero,
                     text="Telecom Customer Churn Predictor",
                     font=("Inter", 22, "bold"), text_color="#ffffff").pack(
            anchor="w", padx=20, pady=(4, 0))

        ctk.CTkLabel(hero,
                     text="Enter customer details below · Random Forest Model · GROUP 1",
                     font=("Inter", 12), text_color="#c4b5fd").pack(
            anchor="w", padx=20, pady=(2, 12))

        # ── Scrollable main content ───────────────────────────────────────
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent",
                                        scrollbar_button_color="#4c1d95",
                                        scrollbar_button_hover_color="#6d28d9")
        scroll.pack(fill="both", expand=True, padx=20, pady=6)

        # Two columns
        left  = ctk.CTkFrame(scroll, fg_color="transparent")
        right = ctk.CTkFrame(scroll, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        scroll.grid_columnconfigure(0, weight=1)
        scroll.grid_columnconfigure(1, weight=1)

        self._build_left(left)
        self._build_right(right)

        # ── Predict button ────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(4, 6))
        self.predict_btn = ctk.CTkButton(
            btn_frame,
            text="🔍   Predict Churn",
            font=("Inter", 15, "bold"),
            fg_color="#6d28d9", hover_color="#5b21b6",
            text_color="#ffffff",
            corner_radius=12, height=48,
            command=self._predict,
        )
        self.predict_btn.pack(fill="x")

        # ── Result area ───────────────────────────────────────────────────
        self.result_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.result_frame.pack(fill="x", padx=20, pady=(0, 16))

    # ── LEFT COLUMN ────────────────────────────────────────────────────────
    def _build_left(self, parent):

        # Personal Info
        card, inner = section_frame(parent, "👤  Personal Information")
        card.pack(fill="x", pady=(0, 10))
        r0 = ctk.CTkFrame(inner, fg_color="transparent")
        r0.pack(fill="x")
        r1 = ctk.CTkFrame(inner, fg_color="transparent")
        r1.pack(fill="x", pady=(6, 0))

        for col in (r0, r1):
            col.grid_columnconfigure(0, weight=1)
            col.grid_columnconfigure(1, weight=1)

        make_label(r0, "Gender").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.gender = make_dropdown(r0, ["Male", "Female"])
        self.gender.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        make_label(r0, "Senior Citizen").grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.senior = make_dropdown(r0, ["No", "Yes"])
        self.senior.grid(row=1, column=1, sticky="ew")

        make_label(r1, "Has Partner").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.partner = make_dropdown(r1, ["No", "Yes"])
        self.partner.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        make_label(r1, "Has Dependents").grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.dependents = make_dropdown(r1, ["No", "Yes"])
        self.dependents.grid(row=1, column=1, sticky="ew")

        # Account Details
        card2, inner2 = section_frame(parent, "📋  Account Details")
        card2.pack(fill="x", pady=(0, 10))

        make_label(inner2, "Tenure (months)").pack(anchor="w", pady=(0, 2))
        self.tenure = make_entry(inner2, placeholder="0 – 72", width=None)
        self.tenure.pack(fill="x")
        self.tenure.insert(0, "12")

        row_ct = ctk.CTkFrame(inner2, fg_color="transparent")
        row_ct.pack(fill="x", pady=(8, 0))
        row_ct.grid_columnconfigure(0, weight=1)
        row_ct.grid_columnconfigure(1, weight=1)

        make_label(row_ct, "Contract Type").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.contract = make_dropdown(row_ct, ["Month-to-month", "One year", "Two year"])
        self.contract.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        make_label(row_ct, "Paperless Billing").grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.paperless = make_dropdown(row_ct, ["No", "Yes"])
        self.paperless.grid(row=1, column=1, sticky="ew")

        make_label(inner2, "Payment Method").pack(anchor="w", pady=(8, 2))
        self.payment = make_dropdown(inner2, [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], width=None)
        self.payment.pack(fill="x")

        # Charges
        card3, inner3 = section_frame(parent, "💰  Charges")
        card3.pack(fill="x", pady=(0, 10))

        row_ch = ctk.CTkFrame(inner3, fg_color="transparent")
        row_ch.pack(fill="x")
        row_ch.grid_columnconfigure(0, weight=1)
        row_ch.grid_columnconfigure(1, weight=1)

        make_label(row_ch, "Monthly Charges ($)").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.monthly = make_entry(row_ch, placeholder="e.g. 65.00")
        self.monthly.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.monthly.insert(0, "65.00")

        make_label(row_ch, "Total Charges ($)").grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.total = make_entry(row_ch, placeholder="e.g. 780.00")
        self.total.grid(row=1, column=1, sticky="ew")
        self.total.insert(0, "780.00")

    # ── RIGHT COLUMN ───────────────────────────────────────────────────────
    def _build_right(self, parent):

        # Phone Services
        card, inner = section_frame(parent, "📞  Phone Services")
        card.pack(fill="x", pady=(0, 10))

        row_ph = ctk.CTkFrame(inner, fg_color="transparent")
        row_ph.pack(fill="x")
        row_ph.grid_columnconfigure(0, weight=1)
        row_ph.grid_columnconfigure(1, weight=1)

        make_label(row_ph, "Phone Service").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.phone = make_dropdown(row_ph, ["Yes", "No"])
        self.phone.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        make_label(row_ph, "Multiple Lines").grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.multi = make_dropdown(row_ph, ["No", "Yes", "No phone service"])
        self.multi.grid(row=1, column=1, sticky="ew")

        # Internet Services
        card2, inner2 = section_frame(parent, "🌐  Internet Services")
        card2.pack(fill="x", pady=(0, 10))

        make_label(inner2, "Internet Service").pack(anchor="w", pady=(0, 2))
        self.internet = make_dropdown(inner2, ["DSL", "Fiber optic", "No"], width=None)
        self.internet.pack(fill="x")

        row_i1 = ctk.CTkFrame(inner2, fg_color="transparent")
        row_i1.pack(fill="x", pady=(8, 0))
        row_i2 = ctk.CTkFrame(inner2, fg_color="transparent")
        row_i2.pack(fill="x", pady=(6, 0))
        row_i3 = ctk.CTkFrame(inner2, fg_color="transparent")
        row_i3.pack(fill="x", pady=(6, 0))

        for row in (row_i1, row_i2, row_i3):
            row.grid_columnconfigure(0, weight=1)
            row.grid_columnconfigure(1, weight=1)

        svc_opts = ["No", "Yes", "No internet service"]

        make_label(row_i1, "Online Security").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.sec = make_dropdown(row_i1, svc_opts)
        self.sec.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        make_label(row_i1, "Online Backup").grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.backup = make_dropdown(row_i1, svc_opts)
        self.backup.grid(row=1, column=1, sticky="ew")

        make_label(row_i2, "Device Protection").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.device = make_dropdown(row_i2, svc_opts)
        self.device.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        make_label(row_i2, "Tech Support").grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.tech = make_dropdown(row_i2, svc_opts)
        self.tech.grid(row=1, column=1, sticky="ew")

        make_label(row_i3, "Streaming TV").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.tv = make_dropdown(row_i3, svc_opts)
        self.tv.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        make_label(row_i3, "Streaming Movies").grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.movies = make_dropdown(row_i3, svc_opts)
        self.movies.grid(row=1, column=1, sticky="ew")

    # ── Prediction ─────────────────────────────────────────────────────────
    def _predict(self):
        # Check model loaded
        if self._model is None:
            if self._model_error:
                messagebox.showerror("Model Error", self._model_error)
            else:
                messagebox.showinfo("Please wait", "Model is still loading. Try again in a moment.")
            return

        # Validate & parse inputs
        try:
            tenure_val = int(self.tenure.get())
            if not (0 <= tenure_val <= 72):
                raise ValueError("Tenure must be between 0 and 72.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Tenure: {e}")
            return

        try:
            monthly_val = float(self.monthly.get())
            total_val   = float(self.total.get())
        except ValueError:
            messagebox.showerror("Input Error", "Charges must be valid numbers.")
            return

        input_data = pd.DataFrame([{
            "gender":            self.gender.get(),
            "SeniorCitizen":     1 if self.senior.get() == "Yes" else 0,
            "Partner":           self.partner.get(),
            "Dependents":        self.dependents.get(),
            "tenure":            tenure_val,
            "PhoneService":      self.phone.get(),
            "MultipleLines":     self.multi.get(),
            "InternetService":   self.internet.get(),
            "OnlineSecurity":    self.sec.get(),
            "OnlineBackup":      self.backup.get(),
            "DeviceProtection":  self.device.get(),
            "TechSupport":       self.tech.get(),
            "StreamingTV":       self.tv.get(),
            "StreamingMovies":   self.movies.get(),
            "Contract":          self.contract.get(),
            "PaperlessBilling":  self.paperless.get(),
            "PaymentMethod":     self.payment.get(),
            "MonthlyCharges":    monthly_val,
            "TotalCharges":      total_val,
        }])

        prediction = self._model.predict(input_data)[0]
        proba      = self._model.predict_proba(input_data)[0]
        churn_prob = proba[1]
        safe_prob  = proba[0]

        self._show_result(prediction, churn_prob, safe_prob, tenure_val,
                          monthly_val, input_data)

    # ── Result display ─────────────────────────────────────────────────────
    def _show_result(self, prediction, churn_prob, safe_prob, tenure_val,
                     monthly_val, input_data):
        # Clear previous result
        for w in self.result_frame.winfo_children():
            w.destroy()

        is_churn   = prediction == 1
        pct        = int((churn_prob if is_churn else safe_prob) * 100)
        accent     = "#ef4444" if is_churn else "#10b981"
        bg_color   = "#450a0a" if is_churn else "#022c22"
        icon       = "⚠️" if is_churn else "✅"
        label_txt  = "HIGH CHURN RISK" if is_churn else "LOW CHURN RISK"
        sub_txt    = "This customer is likely to leave" if is_churn \
                     else "This customer is likely to stay"

        # ── Two-column result layout ──────────────────────────────────────
        res_left  = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        res_right = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        res_left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        res_right.pack(side="left", fill="both", expand=True)

        # Result card
        card = ctk.CTkFrame(res_left, fg_color=bg_color,
                            corner_radius=14, border_width=2,
                            border_color=accent)
        card.pack(fill="both", expand=True)

        ctk.CTkLabel(card, text=icon, font=("Inter", 36)).pack(pady=(18, 0))
        ctk.CTkLabel(card, text=label_txt,
                     font=("Inter", 18, "bold"),
                     text_color=accent).pack()
        ctk.CTkLabel(card, text=sub_txt,
                     font=("Inter", 12),
                     text_color="#94a3b8").pack(pady=(2, 10))

        # Progress bar
        bar_frame = ctk.CTkFrame(card, fg_color="transparent")
        bar_frame.pack(fill="x", padx=24, pady=(0, 4))
        ctk.CTkProgressBar(bar_frame, progress_color=accent,
                           fg_color="#1e293b", height=10,
                           corner_radius=5).pack(fill="x")
        bar_frame.winfo_children()[0].set(pct / 100)

        ctk.CTkLabel(card, text=f"{pct}%",
                     font=("Inter", 28, "bold"),
                     text_color=accent).pack()
        ctk.CTkLabel(card,
                     text="Churn Probability" if is_churn else "Retention Probability",
                     font=("Inter", 11), text_color="#64748b").pack(pady=(0, 10))

        # Probability breakdown
        prob_row = ctk.CTkFrame(card, fg_color="transparent")
        prob_row.pack(fill="x", padx=24, pady=(0, 16))
        prob_row.grid_columnconfigure(0, weight=1)
        prob_row.grid_columnconfigure(1, weight=1)

        for col_idx, (lbl, val, col) in enumerate([
            ("Will NOT Churn", f"{safe_prob*100:.1f}%",  "#10b981"),
            ("Will Churn",     f"{churn_prob*100:.1f}%", "#ef4444"),
        ]):
            box = ctk.CTkFrame(prob_row,
                               fg_color="#0f172a", corner_radius=8,
                               border_width=1, border_color=col)
            box.grid(row=0, column=col_idx, sticky="ew",
                     padx=(0, 4) if col_idx == 0 else (4, 0))
            ctk.CTkLabel(box, text=lbl, font=("Inter", 10, "bold"),
                         text_color=col).pack(pady=(8, 0))
            ctk.CTkLabel(box, text=val, font=("Inter", 16, "bold"),
                         text_color=col).pack(pady=(0, 8))

        # ── Insights card ─────────────────────────────────────────────────
        ins_card = ctk.CTkFrame(res_right, fg_color="#1e1b4b",
                                corner_radius=14, border_width=1,
                                border_color="#312e81")
        ins_card.pack(fill="both", expand=True)

        ctk.CTkLabel(ins_card, text="🔍  Risk Factor Insights",
                     font=("Inter", 12, "bold"),
                     text_color="#a78bfa").pack(anchor="w", padx=14, pady=(12, 0))
        sep = ctk.CTkFrame(ins_card, height=1, fg_color="#312e81")
        sep.pack(fill="x", padx=14, pady=(4, 8))

        insights = self._build_insights(
            tenure_val, monthly_val,
            self.contract.get(), self.internet.get(),
            self.payment.get(), self.senior.get(),
            self.partner.get(), self.dependents.get(),
            self.sec.get(), self.tech.get(),
        )

        for icon_s, text, color in insights:
            row = ctk.CTkFrame(ins_card, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=2)
            ctk.CTkLabel(row, text=icon_s, font=("Inter", 13), width=22).pack(side="left")
            ctk.CTkLabel(row, text=text, font=("Inter", 12),
                         text_color=color, wraplength=320,
                         anchor="w", justify="left").pack(side="left", fill="x", expand=True)

        # Recommendation
        ctk.CTkLabel(ins_card, text="💡  Recommended Action",
                     font=("Inter", 12, "bold"),
                     text_color="#a78bfa").pack(anchor="w", padx=14, pady=(10, 0))
        sep2 = ctk.CTkFrame(ins_card, height=1, fg_color="#312e81")
        sep2.pack(fill="x", padx=14, pady=(4, 6))

        rec, rec_color = self._recommendation(is_churn, churn_prob, safe_prob)
        rec_box = ctk.CTkFrame(ins_card, fg_color="#0f172a", corner_radius=8,
                               border_width=0)
        rec_box.pack(fill="x", padx=14, pady=(0, 14))
        ctk.CTkLabel(rec_box, text=rec, font=("Inter", 12),
                     text_color=rec_color, wraplength=340,
                     anchor="w", justify="left").pack(padx=12, pady=10)

    # ── Insights logic ─────────────────────────────────────────────────────
    def _build_insights(self, tenure, monthly, contract, internet,
                        payment, senior, partner, dependents, security, tech):
        insights = []
        R, G, A = "#f87171", "#34d399", "#fbbf24"

        if tenure <= 6:
            insights.append(("🔴", f"New customer (≤6 months) — higher churn risk", R))
        elif tenure >= 36:
            insights.append(("🟢", f"Loyal customer ({tenure} months tenure)", G))
        else:
            insights.append(("🟡", f"Moderate tenure ({tenure} months)", A))

        if contract == "Month-to-month":
            insights.append(("🔴", "Month-to-month contract — most churn-prone", R))
        elif contract == "One year":
            insights.append(("🟡", "One-year contract — moderate retention", A))
        else:
            insights.append(("🟢", "Two-year contract — high retention", G))

        if internet == "Fiber optic":
            insights.append(("🔴", "Fiber optic — associated with higher churn", R))
        elif internet == "DSL":
            insights.append(("🟡", "DSL service — moderate churn tendency", A))
        else:
            insights.append(("🟢", "No internet service — lower churn risk", G))

        if payment == "Electronic check":
            insights.append(("🔴", "Electronic check — highest churn payment type", R))
        else:
            insights.append(("🟢", "Automatic/check payment — lower churn tendency", G))

        if senior == "Yes":
            insights.append(("🔴", "Senior citizen — tends to have higher churn", R))

        if partner == "No" and dependents == "No":
            insights.append(("🔴", "No partner or dependents — more likely to churn", R))
        else:
            insights.append(("🟢", "Has partner/dependents — lower churn tendency", G))

        if security == "No" and tech == "No":
            insights.append(("🔴", "No online security or tech support", R))
        elif security == "Yes" and tech == "Yes":
            insights.append(("🟢", "Has security & tech support services", G))

        if monthly > 70:
            insights.append(("🔴", f"High monthly charges (${monthly:.0f})", R))
        elif monthly < 30:
            insights.append(("🟢", f"Low monthly charges (${monthly:.0f})", G))
        else:
            insights.append(("🟡", f"Moderate monthly charges (${monthly:.0f})", A))

        return insights

    def _recommendation(self, is_churn, churn_prob, safe_prob):
        if is_churn:
            if churn_prob >= 0.75:
                return ("🚨 Urgent: Offer immediate retention incentives — "
                        "discount, contract upgrade, or dedicated support call.", "#ef4444")
            elif churn_prob >= 0.5:
                return ("⚠️ Proactive: Schedule a customer satisfaction review. "
                        "Consider a loyalty discount or service tier upgrade.", "#fbbf24")
            else:
                return ("👀 Monitor: Mild churn signals. Follow up with a "
                        "satisfaction survey and highlight service benefits.", "#fbbf24")
        else:
            if safe_prob >= 0.80:
                return ("✅ Retain & Upsell: Highly loyal customer. Great candidate "
                        "for premium upgrades or referral programs.", "#34d399")
            else:
                return ("✅ Engage: Customer is likely to stay. Maintain regular "
                        "engagement and ensure service satisfaction.", "#34d399")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = ChurnApp()
    app.mainloop()
