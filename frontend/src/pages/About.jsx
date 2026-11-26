export default function About() {
  return (
    <section className="card">
      <h2>About Indic Review &amp; Sentiment Analyzer</h2>
      <p>Customers write reviews in many Indian languages, and often mix Hindi with English. This tool reads those reviews and tells you whether each one is positive or negative, how confident it is, and which words drove the decision. Business owners can also break a single review down into aspects such as price, quality or delivery, and compare how the different models judge the same review.</p>
      <p>Supported languages: Hindi, Tamil, Telugu, Bengali, Marathi and Hinglish (Hindi written in English letters). The language is detected automatically.</p>
      <h3>How it decides</h3>
      <p>Every review is scored by <b>MuRIL</b>, a transformer model pretrained by Google on 17 Indian languages and fine-tuned here on product reviews. It understands context, not just keywords, and labels about 93% of unseen test reviews correctly.</p>
      <h3>Other models you can compare</h3>
      <ul>
        <li><b>Logistic Regression</b> on word and character patterns (TF-IDF): fast and easy to explain</li>
        <li><b>Multinomial Naive Bayes</b> and <b>Bernoulli Naive Bayes</b>: simple probabilistic models on the same patterns</li>
        <li><b>BiLSTM</b>: a neural network that reads each review in both directions</li>
      </ul>
      <p>Aspect extraction works offline: the review is split into parts, aspect words (camera, battery, price, delivery…) are recognised in all six languages, and the trained model judges each part.</p>
      <h3>Data</h3>
      <p>The models learned from AI4Bharat IndicSentiment, a public set of product reviews in Indian languages, plus Hinglish reviews created from it. Every translation of a review stays on the same side of the train/test split, so the test scores are not inflated by the model having seen the same review in another language.</p>
      <h3>Good to know</h3>
      <p>Very short, casual Hinglish praise such as "bahut accha hai" can still be misread, because the training data contains few reviews written like that. Everything runs on this server; no review text is sent anywhere.</p>
    </section>
  )
}
