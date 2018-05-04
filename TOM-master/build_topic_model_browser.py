# coding: utf-8
import os
import shutil
import tom_lib.utils as utils
from flask import Flask, render_template
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization, LatentDirichletAllocation, TruncatedSVD
from tom_lib.structure.corpus import Corpus
import pandas as pd

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

# Flask Web server
app = Flask(__name__, static_folder='browser/static', template_folder='browser/templates')

# Parameters
max_tf = 0.8
min_tf = 4
num_topics = 30
vectorization = 'tfidf'

# Load corpus
corpus = Corpus(source_file_path='input/egc_lemmatized_.csv',
                language='english',
                vectorization=vectorization,
                max_relative_frequency=max_tf,
                min_absolute_frequency=min_tf)
print('corpus size:', corpus.size)
print('vocabulary size:', len(corpus.vocabulary))

# Infer topics
topic_model = NonNegativeMatrixFactorization(corpus=corpus)
topic_model.infer_topics(num_topics=num_topics)
topic_model.print_topics(num_words=30)

# Clean the data directory
if os.path.exists('browser/static/data'):
    shutil.rmtree('browser/static/data')
os.makedirs('browser/static/data')

# Export topic cloud
utils.save_topic_cloud(topic_model, 'browser/static/data/topic_cloud.json')


# Export details about topics
for topic_id in range(topic_model.nb_topics):
    utils.save_word_distribution(topic_model.top_words(topic_id, 50),
                                 'browser/static/data/word_distribution' + str(topic_id) + '.tsv')
    utils.save_affiliation_repartition(topic_model.affiliation_repartition(topic_id),
                                       'browser/static/data/affiliation_repartition' + str(topic_id) + '.tsv')
    evolution = []
    for i in range(2012, 2016):
        evolution.append((i, topic_model.topic_frequency(topic_id, date=i)))
    utils.save_topic_evolution(evolution, 'browser/static/data/frequency' + str(topic_id) + '.tsv')

# Export details about documents
for doc_id in range(topic_model.corpus.size):
    utils.save_topic_distribution(topic_model.topic_distribution_for_document(doc_id),
                                  'browser/static/data/topic_distribution_d' + str(doc_id) + '.tsv')

# Export details about words
for word_id in range(len(topic_model.corpus.vocabulary)):
    utils.save_topic_distribution(topic_model.topic_distribution_for_word(word_id),
                                  'browser/static/data/topic_distribution_w' + str(word_id) + '.tsv')

# Associate documents with topics
topic_associations = topic_model.documents_per_topic()

# Export per-topic author network
#for topic_id in range(topic_model.nb_topics):
#    utils.save_json_object(corpus.collaboration_network(topic_associations[topic_id]),
#                           'browser/static/data/author_network' + str(topic_id) + '.json')


@app.route('/')
def index():
    return render_template('start.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           method=type(topic_model).__name__,
                           corpus_size=corpus.size,
                           vocabulary_size=len(corpus.vocabulary),
                           max_tf=max_tf,
                           min_tf=min_tf,
                           vectorization=vectorization,
                           num_topics=num_topics)


@app.route('/topic_cloud.html')
def topic_cloud():
    return render_template('topic_cloud.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/vocabulary.html')
def vocabulary():
    word_list = []
    for i in range(len(corpus.vocabulary)):
        word_list.append((i, corpus.word_for_id(i)))
    splitted_vocabulary = []
    words_per_column = int(len(corpus.vocabulary)/5)
    for j in range(5):
        sub_vocabulary = []
        for l in range(j*words_per_column, (j+1)*words_per_column):
            sub_vocabulary.append(word_list[l])
        splitted_vocabulary.append(sub_vocabulary)
    return render_template('vocabulary.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           splitted_vocabulary=splitted_vocabulary,
                           vocabulary_size=len(word_list))


@app.route('/topic/<tid>.html')
def topic_details(tid):
    ids = topic_associations[int(tid)]
    documents = []
    title_count = []
  

    for document_id in ids:
        documents.append((corpus.title(document_id).capitalize(),
                          ', '.join(corpus.author(document_id)),
                          corpus.date(document_id), document_id))
    import pandas as pd

    labels = ['document_id', 'description', 'date','doc']
    df = pd.DataFrame.from_records(documents, columns=labels)
    print(df.head())
    print(df.document_id.value_counts())
    df.document_id.value_counts().values.tolist()
    s = df.document_id.value_counts()
    for i, v in s.iteritems():
      title_count.append((i,v))

    return render_template('topic.html',
                           topic_id=tid,
                           frequency=round(topic_model.topic_frequency(int(tid))*100, 2),
                           documents=documents,
                           title_count=title_count,
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/document/<did>.html')
def document_details(did):
    vector = topic_model.corpus.vector_for_document(int(did))
    word_list = []
    for a_word_id in range(len(vector)):
        word_list.append((corpus.word_for_id(a_word_id), round(vector[a_word_id], 3), a_word_id))
    word_list.sort(key=lambda x: x[1])
    word_list.reverse()
    documents = []
    for another_doc in corpus.similar_documents(int(did), 5):
        documents.append((corpus.title(another_doc[0]).capitalize(),
                          ', '.join(corpus.author(another_doc[0])),
                          corpus.date(another_doc[0]), another_doc[0], round(another_doc[1], 3)))
    return render_template('document.html',
                           doc_id=did,
                           words=word_list[:21],
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           documents=documents,
                           authors=', '.join(corpus.author(int(did))),
                           year=corpus.date(int(did)),
                           short_content=corpus.title(int(did)))


@app.route('/word/<wid>.html')
def word_details(wid):
    documents = []
    for document_id in corpus.docs_for_word(int(wid)):
        documents.append((corpus.title(document_id).capitalize(),
                          ', '.join(corpus.author(document_id)),
                          corpus.date(document_id), document_id))
    return render_template('word.html',
                           word_id=wid,
                           word=topic_model.corpus.word_for_id(int(wid)),
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           documents=documents)




##################################################################################################
print("lda---------------------------------------------------------------------------------------")

# Infer topics
topic_model_lda = LatentDirichletAllocation(corpus=corpus)
topic_model_lda.infer_topics(num_topics=num_topics)
topic_model_lda.print_topics(num_words=30)

# Clean the data directory
if os.path.exists('browser/static/data_lda'):
    shutil.rmtree('browser/static/data_lda')
os.makedirs('browser/static/data_lda')

# Export topic cloud
utils.save_topic_cloud(topic_model_lda, 'browser/static/data_lda/topic_cloud.json')

# Export details about topics
for topic_id in range(topic_model_lda.nb_topics):
    utils.save_word_distribution(topic_model_lda.top_words(topic_id, 50),
                                 'browser/static/data_lda/word_distribution' + str(topic_id) + '.tsv')
    utils.save_affiliation_repartition(topic_model_lda.affiliation_repartition(topic_id),
                                       'browser/static/data_lda/affiliation_repartition' + str(topic_id) + '.tsv')
    evolution = []
    for i in range(2012, 2016):
        evolution.append((i, topic_model_lda.topic_frequency(topic_id, date=i)))
    utils.save_topic_evolution(evolution, 'browser/static/data_lda/frequency' + str(topic_id) + '.tsv')

# Export details about documents
for doc_id in range(topic_model_lda.corpus.size):
    utils.save_topic_distribution(topic_model_lda.topic_distribution_for_document(doc_id),
                                  'browser/static/data_lda/topic_distribution_d' + str(doc_id) + '.tsv')

# Export details about words
for word_id in range(len(topic_model_lda.corpus.vocabulary)):
    utils.save_topic_distribution(topic_model_lda.topic_distribution_for_word(word_id),
                                  'browser/static/data_lda/topic_distribution_w' + str(word_id) + '.tsv')

# Associate documents with topics
topic_associations = topic_model_lda.documents_per_topic()

# Export per-topic author network
#for topic_id in range(topic_model.nb_topics):
#    utils.save_json_object(corpus.collaboration_network(topic_associations[topic_id]),
#                           'browser/static/data/author_network' + str(topic_id) + '.json')


@app.route('/lda')
def index_lda():
    return render_template('index_lda.html',
                           topic_ids=range(topic_model_lda.nb_topics),
                           doc_ids=range(corpus.size),
                           method=type(topic_model_lda).__name__,
                           corpus_size=corpus.size,
                           vocabulary_size=len(corpus.vocabulary),
                           max_tf=max_tf,
                           min_tf=min_tf,
                           vectorization=vectorization,
                           num_topics=num_topics)


@app.route('/topic_cloud_lda.html')
def topic_cloud_lda():
    return render_template('topic_cloud_lda.html',
                           topic_ids=range(topic_model_lda.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/vocabulary_lda.html')
def vocabulary_lda():
    word_list = []
    for i in range(len(corpus.vocabulary)):
        word_list.append((i, corpus.word_for_id(i)))
    splitted_vocabulary = []
    words_per_column = int(len(corpus.vocabulary)/5)
    for j in range(5):
        sub_vocabulary = []
        for l in range(j*words_per_column, (j+1)*words_per_column):
            sub_vocabulary.append(word_list[l])
        splitted_vocabulary.append(sub_vocabulary)
    return render_template('vocabulary_lda.html',
                           topic_ids=range(topic_model_lda.nb_topics),
                           doc_ids=range(corpus.size),
                           splitted_vocabulary=splitted_vocabulary,
                           vocabulary_size=len(word_list))


@app.route('/topic_lda/<tid>.html')
def topic_details_lda(tid):
    ids = topic_associations[int(tid)]
    documents = []
    title_count = []
  

    for document_id in ids:
        documents.append((corpus.title(document_id).capitalize(),
                          ', '.join(corpus.author(document_id)),
                          corpus.date(document_id), document_id))
    import pandas as pd

    labels = ['document_id', 'description', 'date','doc']
    df = pd.DataFrame.from_records(documents, columns=labels)
    print(df.head())
    print(df.document_id.value_counts())
    df.document_id.value_counts().values.tolist()
    s = df.document_id.value_counts()
    for i, v in s.iteritems():
      title_count.append((i,v))

    return render_template('topic_lda.html',
                           topic_id=tid,
                           frequency=round(topic_model_lda.topic_frequency(int(tid))*100, 2),
                           documents=documents,
                           title_count=title_count,
                           topic_ids=range(topic_model_lda.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/document_lda/<did>.html')
def document_details_lda(did):
    vector = topic_model_lda.corpus.vector_for_document(int(did))
    word_list = []
    for a_word_id in range(len(vector)):
        word_list.append((corpus.word_for_id(a_word_id), round(vector[a_word_id], 3), a_word_id))
    word_list.sort(key=lambda x: x[1])
    word_list.reverse()
    documents = []
    for another_doc in corpus.similar_documents(int(did), 5):
        documents.append((corpus.title(another_doc[0]).capitalize(),
                          ', '.join(corpus.author(another_doc[0])),
                          corpus.date(another_doc[0]), another_doc[0], round(another_doc[1], 3)))
    return render_template('document_lda.html',
                           doc_id=did,
                           words=word_list[:21],
                           topic_ids=range(topic_model_lda.nb_topics),
                           doc_ids=range(corpus.size),
                           documents=documents,
                           authors=', '.join(corpus.author(int(did))),
                           year=corpus.date(int(did)),
                           short_content=corpus.title(int(did)))


@app.route('/word_lda/<wid>.html')
def word_details_lda(wid):
    documents = []
    for document_id in corpus.docs_for_word(int(wid)):
        documents.append((corpus.title(document_id).capitalize(),
                          ', '.join(corpus.author(document_id)),
                          corpus.date(document_id), document_id))
    return render_template('word_lda.html',
                           word_id=wid,
                           word=topic_model_lda.corpus.word_for_id(int(wid)),
                           topic_ids=range(topic_model_lda.nb_topics),
                           doc_ids=range(corpus.size),
                           documents=documents)



#########################################################################################################    


print("svd---------------------------------------------------------------------------------------")

# Infer topics
topic_model_svd = TruncatedSVD(corpus=corpus)
topic_model_svd.infer_topics(num_topics=num_topics)
topic_model_svd.print_topics(num_words=30)

# Clean the data directory
if os.path.exists('browser/static/data_svd'):
    shutil.rmtree('browser/static/data_svd')
os.makedirs('browser/static/data_svd')

# Export topic cloud
utils.save_topic_cloud(topic_model_svd, 'browser/static/data_svd/topic_cloud.json')

# Export details about topics
for topic_id in range(topic_model_svd.nb_topics):
    utils.save_word_distribution(topic_model_svd.top_words(topic_id, 50),
                                 'browser/static/data_svd/word_distribution' + str(topic_id) + '.tsv')
    utils.save_affiliation_repartition(topic_model_svd.affiliation_repartition(topic_id),
                                       'browser/static/data_svd/affiliation_repartition' + str(topic_id) + '.tsv')
    evolution = []
    for i in range(2012, 2016):
        evolution.append((i, topic_model_svd.topic_frequency(topic_id, date=i)))
    utils.save_topic_evolution(evolution, 'browser/static/data_svd/frequency' + str(topic_id) + '.tsv')

# Export details about documents
for doc_id in range(topic_model_svd.corpus.size):
    utils.save_topic_distribution(topic_model_svd.topic_distribution_for_document(doc_id),
                                  'browser/static/data_svd/topic_distribution_d' + str(doc_id) + '.tsv')

# Export details about words
for word_id in range(len(topic_model_svd.corpus.vocabulary)):
    utils.save_topic_distribution(topic_model_svd.topic_distribution_for_word(word_id),
                                  'browser/static/data_svd/topic_distribution_w' + str(word_id) + '.tsv')

# Associate documents with topics
topic_associations = topic_model_svd.documents_per_topic()

# Export per-topic author network
#for topic_id in range(topic_model.nb_topics):
#    utils.save_json_object(corpus.collaboration_network(topic_associations[topic_id]),
#                           'browser/static/data/author_network' + str(topic_id) + '.json')


@app.route('/svd')
def index_svd():
    return render_template('index_svd.html',
                           topic_ids=range(topic_model_svd.nb_topics),
                           doc_ids=range(corpus.size),
                           method=type(topic_model_svd).__name__,
                           corpus_size=corpus.size,
                           vocabulary_size=len(corpus.vocabulary),
                           max_tf=max_tf,
                           min_tf=min_tf,
                           vectorization=vectorization,
                           num_topics=num_topics)


@app.route('/topic_cloud_svd.html')
def topic_cloud_svd():
    return render_template('topic_cloud_svd.html',
                           topic_ids=range(topic_model_svd.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/vocabulary_svd.html')
def vocabulary_svd():
    word_list = []
    for i in range(len(corpus.vocabulary)):
        word_list.append((i, corpus.word_for_id(i)))
    splitted_vocabulary = []
    words_per_column = int(len(corpus.vocabulary)/5)
    for j in range(5):
        sub_vocabulary = []
        for l in range(j*words_per_column, (j+1)*words_per_column):
            sub_vocabulary.append(word_list[l])
        splitted_vocabulary.append(sub_vocabulary)
    return render_template('vocabulary_svd.html',
                           topic_ids=range(topic_model_svd.nb_topics),
                           doc_ids=range(corpus.size),
                           splitted_vocabulary=splitted_vocabulary,
                           vocabulary_size=len(word_list))


@app.route('/topic_svd/<tid>.html')
def topic_details_svd(tid):
    ids = topic_associations[int(tid)]
    documents = []
    title_count = []
  

    for document_id in ids:
        documents.append((corpus.title(document_id).capitalize(),
                          ', '.join(corpus.author(document_id)),
                          corpus.date(document_id), document_id))
    import pandas as pd

    labels = ['document_id', 'description', 'date','doc']
    df = pd.DataFrame.from_records(documents, columns=labels)
    print(df.head())
    print(df.document_id.value_counts())
    df.document_id.value_counts().values.tolist()
    s = df.document_id.value_counts()
    for i, v in s.iteritems():
      title_count.append((i,v))

    return render_template('topic_svd.html',
                           topic_id=tid,
                           frequency=round(topic_model_svd.topic_frequency(int(tid))*100, 2),
                           documents=documents,
                           title_count=title_count,
                           topic_ids=range(topic_model_svd.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/document_svd/<did>.html')
def document_details_svd(did):
    vector = topic_model_svd.corpus.vector_for_document(int(did))
    word_list = []
    for a_word_id in range(len(vector)):
        word_list.append((corpus.word_for_id(a_word_id), round(vector[a_word_id], 3), a_word_id))
    word_list.sort(key=lambda x: x[1])
    word_list.reverse()
    documents = []
    for another_doc in corpus.similar_documents(int(did), 5):
        documents.append((corpus.title(another_doc[0]).capitalize(),
                          ', '.join(corpus.author(another_doc[0])),
                          corpus.date(another_doc[0]), another_doc[0], round(another_doc[1], 3)))
    return render_template('document_svd.html',
                           doc_id=did,
                           words=word_list[:21],
                           topic_ids=range(topic_model_svd.nb_topics),
                           doc_ids=range(corpus.size),
                           documents=documents,
                           authors=', '.join(corpus.author(int(did))),
                           year=corpus.date(int(did)),
                           short_content=corpus.title(int(did)))


@app.route('/word_svd/<wid>.html')
def word_details_svd(wid):
    documents = []
    for document_id in corpus.docs_for_word(int(wid)):
        documents.append((corpus.title(document_id).capitalize(),
                          ', '.join(corpus.author(document_id)),
                          corpus.date(document_id), document_id))
    return render_template('word_svd.html',
                           word_id=wid,
                           word=topic_model_svd.corpus.word_for_id(int(wid)),
                           topic_ids=range(topic_model_svd.nb_topics),
                           doc_ids=range(corpus.size),
                           documents=documents)



#########################################################################################################    


if __name__ == '__main__':
    # Access the browser at http://localhost:2016/
    app.run(debug=True, host='localhost', port=2016
      )

