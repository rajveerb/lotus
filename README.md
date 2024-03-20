<p align="center">
    <img src="P3Tracer.png" width="40%" alt="P3Tracer">
</p>

<p align="center"><i><b>Fine-grained Profiling for Machine Learning Data
Preprocessing in PyTorch</b></i></p>


We introduce **P3Tracer**, a specialized profiling tool for PyTorch preprocessing pipelines. P3Tracer
employs two novel approaches:

1. **P3Torch** - An instrumentation methodology for the PyTorch library, which enables fine-grained elapsed time profiling with minimal time and storage overheads. 
2. **P3Map** - A mapping methodology to reconstruct a mapping between Python functions and the underlying C++ functions they call, effectively linking high-level Python functions with low-level hardware counters. 

Above combination is powerful as it allows enables users to better reason about their pipeline’s performance, both at the level of preprocessing operations and their performance on hardware resource usage.

## Quick Links

- [Quick Links](#quick-links)
- [Documentation](#documentation)
- [Why EvaDB](#why-evadb)
- [How does EvaDB work](#how-does-evadb-work)
- [Illustrative Queries](#illustrative-queries)
- [Illustrative Apps](#illustrative-apps)
- [More Illustrative Queries](#more-illustrative-queries)
- [Architecture of EvaDB](#architecture-of-evadb)
- [Community and Support](#community-and-support)
- [Contributing](#contributing)
- [Star History](#star-history)
- [License](#license)

## Documentation

You can find the complete documentation of EvaDB at [evadb.ai/docs](https://evadb.ai/docs/) 📚✨🚀

## Why EvaDB
 
In the world of AI, we've reached a stage where many AI tasks that were traditionally handled by AI or ML engineers can now be automated. EvaDB enables software developers with the ability to perform advanced AI tasks without needing to delve into the intricate details.

EvaDB covers many AI applications, including regression, classification, image recognition, question answering, and many other generative AI applications. EvaDB targets 99% of AI problems that are often repetitive and can be automated with a simple function call in an SQL query. Until now, there is no comprehensive open-source framework for bringing AI into an existing SQL database system with a principled AI optimization framework, and that's where EvaDB comes in.

Our target audience is software developers who may not necessarily have a background in AI but require AI capabilities to solve specific problems. We target programmers who write simple SQL queries inside their CRUD apps. With EvaDB, it is possible to easily add AI features to these apps by calling built-in AI functions in the queries.

## How does EvaDB work

<details>
<ul>
<li>Connect EvaDB to your SQL and vector database systems with the <a href="https://evadb.readthedocs.io/en/stable/source/reference/databases/postgres.html">`CREATE DATABASE`</a> and <a href="https://evadb.readthedocs.io/en/stable/source/reference/evaql/create_index.html">`CREATE INDEX`</a> statements.</li>
<li>Write SQL queries with AI functions to get inference results:</li>
   <ul>
   <li>Pick a pre-trained AI model from Hugging Face, Open AI, Ultralytics, PyTorch, and built-in AI frameworks for generative AI, NLP, and vision applications;</li>  
   <li>or pick from a variety of state-of-the-art ML engines for classic ML use-cases (classification, regression, etc.);</li>
   <li>or bring your custom model built with any AI/ML framework using `CREATE FUNCTION`.</li>
   </ul>
</ul> 
  
Follow the [getting started](https://evadb.readthedocs.io/en/stable/source/overview/getting-started.html) guide to get on-boarded as fast as possible.
</details>

## Illustrative Queries

* Get insights about Github stargazers using GPT4.

```sql
SELECT name, country, email, programming_languages, social_media, GPT4(prompt,topics_of_interest)
FROM gpt4all_StargazerInsights;

--- Prompt to GPT-4
You are given 10 rows of input, each row is separated by two new line characters.
Categorize the topics listed in each row into one or more of the following 3 technical areas - Machine Learning, Databases, and Web development. If the topics listed are not related to any of these 3 areas, output a single N/A. Do not miss any input row. Do not add any additional text or numbers to your output.
The output rows must be separated by two new line characters. Each input row must generate exactly one output row. For example, the input row [Recommendation systems, Deep neural networks, Postgres] must generate only the output row [Machine Learning, Databases].
The input row [enterpreneurship, startups, venture capital] must generate the output row N/A.
```

* Build a vector index on the feature embeddings returned by the SIFT Feature Extractor on a collection of Reddit images. Return the top-5 similar images for a given image.

```sql
CREATE INDEX reddit_sift_image_index
    ON reddit_dataset (SiftFeatureExtractor(data))
    USING FAISS

SELECT name FROM reddit_dataset ORDER BY
    Similarity(
        SiftFeatureExtractor(Open('reddit-images/g1074_d4mxztt.jpg')),
        SiftFeatureExtractor(data)
    )
    LIMIT 5
```

## Illustrative Apps

Here are some illustrative AI apps built using EvaDB (each notebook can be opened on Google Colab):

 * 🔮 <a href="https://evadb.readthedocs.io/en/stable/source/usecases/sentiment-analysis.html">Sentiment Analysis using LLM within PostgreSQL</a>
 * 🔮 <a href="https://evadb.readthedocs.io/en/stable/source/usecases/question-answering.html">ChatGPT-based Video Question Answering</a>
 * 🔮 <a href="https://evadb.readthedocs.io/en/stable/source/usecases/text-summarization.html">Text Summarization on PDF Documents</a>
 * 🔮 <a href="https://evadb.readthedocs.io/en/stable/source/usecases/object-detection.html">Analysing Traffic Flow with YOLO</a>
 * 🔮 <a href="https://evadb.readthedocs.io/en/stable/source/usecases/emotion-analysis.html">Examining Emotions of Movie</a>
 * 🔮 <a href="https://evadb.readthedocs.io/en/stable/source/usecases/image-search.html">Image Similarity Search</a>


## More Illustrative Queries

<details>

* Get a transcript from a video stored in a table using a Speech Recognition model. Then, ask questions on the extracted transcript using ChatGPT.

```sql
CREATE TABLE text_summary AS
    SELECT SpeechRecognizer(audio) FROM ukraine_video;
SELECT ChatGPT('Is this video summary related to Ukraine russia war', text)
    FROM text_summary;
```

* Train a classic ML model for prediction using the <a href="https://ludwig.ai/latest/">Ludwig AI</a> engine.

```sql
CREATE FUNCTION IF NOT EXISTS PredictHouseRent FROM
(SELECT * FROM HomeRentals)
TYPE Ludwig
PREDICT 'rental_price'
TIME_LIMIT 120;
```

</details>

## Architecture of EvaDB

<details>	
EvaDB's AI-centric query optimizer takes a query as input and generates a query plan. The query engine takes the query plan and hits the relevant backends to efficiently process the query:
1. SQL Database Systems (Structured Data)
2. AI Frameworks (Transform Unstructured Data to Structured Data; Unstructured data includes PDFs, text, images, etc. stored locally or on the cloud)
3. Vector Database Systems (Feature Embeddings)

<p align="center">
  <img width="70%" alt="Architecture Diagram" src="https://raw.githubusercontent.com/georgia-tech-db/evadb/staging/docs/images/evadb/eva-arch-for-user.png">
</p>
</details>

## Community and Support

We would love to learn about your AI app. Please complete this 1-minute form: https://v0fbgcue0cm.typeform.com/to/BZHZWeZm

<!--<p>
  <a href="https://evadb.ai/community">
      <img width="70%" src="https://raw.githubusercontent.com/georgia-tech-db/evadb/master/docs/images/evadb/evadb-slack.png" alt="EvaDB Slack Channel">
  </a>
</p>-->

If you run into any bugs or have any comments, you can reach us on our <a href="https://evadb.ai/community">Slack Community 📟</a>  or create a [Github Issue :bug:](https://github.com/georgia-tech-db/evadb/issues). 

Here is EvaDB's public [roadmap 🛤️](https://github.com/orgs/georgia-tech-db/projects/3). We prioritize features based on user feedback, so we'd love to hear from you!

## Contributing

We are a lean team on a mission to bring AI inside database systems! All kinds of contributions to EvaDB are appreciated 🙌 If you'd like to get involved, here's information on where we could use your help: [contribution guide](https://evadb.readthedocs.io/en/latest/source/dev-guide/contribute.html) 🤗

<p align="center">
  <a href="https://github.com/georgia-tech-db/evadb/graphs/contributors">
    <img width="70%" src="https://contrib.rocks/image?repo=georgia-tech-db/evadb" />
  </a>
</p>

<details>
<b> CI Status: </b> 

[![CI Status](https://circleci.com/gh/georgia-tech-db/evadb.svg?style=svg)](https://circleci.com/gh/georgia-tech-db/evadb)
[![Documentation Status](https://readthedocs.org/projects/evadb/badge/?version=latest)](https://evadb.readthedocs.io/en/latest/index.html)
</details>

## Star History

<p align="center">
  <a href="https://star-history.com/#georgia-tech-db/evadb&Date">
      <img width="90%" src="https://api.star-history.com/svg?repos=georgia-tech-db/evadb&type=Date" alt="EvaDB Star History Chart">
  </a>
</p>

## License
Copyright (c) [Georgia Tech Database Group](http://db.cc.gatech.edu/).
Licensed under an [Apache License](LICENSE.txt).