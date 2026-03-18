from textwrap3 import wrap
import random
import numpy as np
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import os
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import spacy
import string
import pke
import traceback
from flashtext import KeywordProcessor
import numpy as np
from sense2vec import Sense2Vec
script_dir = os.path.dirname(os.path.abspath(__file__))
s2v = Sense2Vec().from_disk(os.path.join(script_dir, 's2v_old'))
from sentence_transformers import SentenceTransformer
import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity




summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)




def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final


def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary




# Loading SpaCy model once (prevents redundant loading)
nlp = spacy.load("en_core_web_sm")

def get_nouns_multipartite(content):
    """Extracts keywords using MultipartiteRank."""
    out = []
    try:
        # Check if content is empty
        if not content.strip():
            return []

        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en', spacy_model=nlp)  # Pass NLP model instance
        
        # Define POS tags to extract
        pos = {'PROPN', 'NOUN'}
        
        # Create stopword list
        stoplist = list(string.punctuation)  # Add punctuation
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']  # Add special tokens
        stoplist += stopwords.words('english')  # Add English stopwords
        
        # Extract noun candidates
        extractor.candidate_selection(pos=pos)

        # Rank keywords using Multipartite graph
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=30)  # Get top n keyphrases

        # Extract keywords
        out = [val[0] for val in keyphrases]

    except Exception:
        traceback.print_exc()
        out = []

    return out

def get_keywords(originaltext, summarytext):
    """Finds important keywords in the summarized text."""
    keywords = get_nouns_multipartite(originaltext)
    print("Keywords unsummarized: ", keywords)

    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    keywords_found = keyword_processor.extract_keywords(summarytext)
    keywords_found = list(set(keywords_found))
    print("Keywords found in summarized: ", keywords_found)

    important_keywords = [keyword for keyword in keywords if keyword in keywords_found]

    return important_keywords[:10]  # Return only top 10  important keywords



# 6th

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)

# 7th

def get_question(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question


# 8th


# paraphrase-distilroberta-base-v1
sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')



def filter_same_sense_words(original, wordlist):
    """Filter words with the same sense as the original word."""
    filtered_words = []
    try:
        base_sense = original.split('|')[1]
        print(f"Base sense: {base_sense}")

        for eachword in wordlist:
            try:
                word, _ = eachword  # Unpack tuple
                if '|' in word and word.split('|')[1] == base_sense:
                    filtered_words.append(word.split('|')[0].replace("_", " ").title().strip())
            except IndexError:
                print(f"Skipping word {eachword} due to incorrect format.")
    except Exception as e:
        print(f"Error in filter_same_sense_words: {e}")

    print(f"Filtered words: {filtered_words}")
    return filtered_words

def get_highest_similarity_score(wordlist, wrd):
    """Find the highest similarity score between a word and a list of words."""
    if not wordlist:
        return 0  # Return 0 if list is empty to prevent errors
    scores = [Levenshtein.normalized_similarity(each.lower(), wrd.lower()) for each in wordlist]
    return max(scores)

def sense2vec_get_words(word, s2v, topn, question):
    """Retrieve similar words using sense2vec."""
    print(f"Word: {word}")
    output = []

    try:
        sense = s2v.get_best_sense(word, senses=[
            "NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT",
            "NORP", "WORK OF ART", "FAC", "GPE", "NUM", "FACILITY"
        ])

        if not sense:
            print(f"No sense found for {word}")
            return []

        most_similar = s2v.most_similar(sense, n=topn)
        print(f"Checking similar words: {most_similar}")

        output = filter_same_sense_words(sense, most_similar)
        print(f"Similar words after filtering: {output}")

    except Exception as e:
        print(f"Error in sense2vec_get_words: {e}")
        return []

    # Apply filtering
    threshold = 0.6
    final = [word]
    checklist = question.split()

    for x in output:
        if get_highest_similarity_score(final, x) < threshold and x not in final and x not in checklist:
            final.append(x)

    print(f"Final words selected: {final[1:]}")
    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    """Maximal Marginal Relevance (MMR) for extracting diverse keywords."""
    if len(words) == 0:
        print("Error: No words provided for MMR selection.")
        return []

    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    if word_doc_similarity.size == 0 or word_similarity.size == 0:
        print("Error: Empty similarity matrices.")
        return []

    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(min(top_n - 1, len(candidates_idx))):  # Prevent out-of-bounds error
        if not candidates_idx:
            break  # Prevent argmax on an empty list

        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        mmr = lambda_param * candidate_similarities - (1 - lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    print(f"Final selected keywords: {[words[idx] for idx in keywords_idx]}")
    return [words[idx] for idx in keywords_idx]



def get_distractors_wordnet(word):
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]
      
      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0: 
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

def get_distractors (word,origsentence,sense2vecmodel,sentencemodel,top_n,lambdaval):
  distractors = sense2vec_get_words(word,sense2vecmodel,top_n,origsentence)
  print ("distractors ",distractors)
  if len(distractors) ==0:
    return distractors
  distractors_new = [word.capitalize()]
  distractors_new.extend(distractors)
  # print ("distractors_new .. ",distractors_new)

  embedding_sentence = origsentence+ " "+word.capitalize()
  # embedding_sentence = word
  keyword_embedding = sentencemodel.encode([embedding_sentence])
  distractor_embeddings = sentencemodel.encode(distractors_new)

  # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors,4,0.7)
  max_keywords = min(len(distractors_new),5)
  filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
  # filtered_keywords = filtered_keywords[1:]
  final = [word.capitalize()]
  for wrd in filtered_keywords:
    if wrd.lower() !=word.lower():
      final.append(wrd.capitalize())
  final = final[1:]
  return final



from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import random
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL if deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    context: str
    method: str  # Wordnet or Sense2Vec

@app.post("/generate")
def generate_question(data: InputData):
    context = data.context
    method = data.method
    
    # Summarize input text
    summary_text = summarizer(context, summary_model, summary_tokenizer)

    # Extract keywords
    np = get_keywords(context, summary_text)

    output = ""
    for answer in np:
        ques = get_question(summary_text, answer, question_model, question_tokenizer)

        # Generate distractors
        if method == "Wordnet":
            distractors = get_distractors_wordnet(answer)
        else:
            distractors = get_distractors(answer.capitalize(), ques, s2v, sentence_transformer_model, 50, 0.2)

        # Shuffle options (correct answer + distractors)
        options = [answer.capitalize()] + distractors[:3]  # Keep top 3 distractors
        random.shuffle(options)

        # Generate MCQ HTML
        options_html = "".join(
            [f"<p>{chr(65 + i)}. {option}</p>" for i, option in enumerate(options)]
        )

        output += f"<div class='mcq'><b class='question'>{ques}</b><br>{options_html}"
        output += f"<b class='answer'>Correct Answer: {answer.capitalize()}</b></div>"

    return {"summary": summary_text, "mcq": output}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
