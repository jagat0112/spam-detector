# 📧 Spam Message Classifier (Naive Bayes + Streamlit)

This is a simple text classification app that predicts whether a given SMS message is **Spam** or **Not Spam**. It was built using **Naive Bayes**, trained on real-world SMS data, and deployed using **Streamlit**.

---

## 🧠 How I Learned This

### 🔢 Step 1: Understanding the Math (Bayes' Theorem)
Before jumping into code, I learned the **Naive Bayes formula** by doing hands-on problems using conditional probability.

I practiced calculating:

- **Prior probability** (e.g. how many people had fever, flu, or COVID)
- **Conditional probability** (e.g. probability of having flu **given** someone had fever)
- Combined it using the Bayes formula:
  
  \[
  P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
  \]

This helped me build strong intuition behind how Naive Bayes works.

---

### 💻 Step 2: Writing the Classifier in Python
Once I understood the concept, I used the **SMS Spam Collection dataset** to train a Naive Bayes model.

What I did:

- Cleaned and labeled SMS data
- Converted text to numbers using **TF-IDF vectorization**
- Trained a `MultinomialNB` model
- Saved both the model and vectorizer using `joblib`

---

### 🌐 Step 3: Building a Web App with Streamlit
I created a simple interface using Streamlit that:

- Lets the user input any SMS message
- Transforms the input with the saved TF-IDF vectorizer
- Predicts if it's spam using the saved Naive Bayes model
- Displays confidence score with 🎯 precision

You can try it with test inputs like:

> 🎉 Congratulations! You’ve won a FREE prize — click now!

---

## 📁 Files in This Repo

```bash
├── app.py                  # Streamlit app
├── spam_model.pkl          # Trained Naive Bayes model
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── requirements.txt        # Python dependencies
