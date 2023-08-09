from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer
from nltk.stem import SnowballStemmer
import numpy as np

model = SentenceTransformer('DeepPavlov/rubert-base-cased')
stemmer = SnowballStemmer("russian")
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

#Upload your answer options
with open(r"Poll-Gen\data.txt", "r", encoding="utf-8") as file:
    data = [line.strip() for line in file]
    print("Data from data.txt loaded")

def ai_poll_generator(question, num_of_options):
    answer_data = []

    data_embeddings = model.encode(data, convert_to_tensor=True)

    user_input_cleaned = question.lower()

    user_input_tokens = tokenizer.tokenize(user_input_cleaned)
    user_input_stems = [stemmer.stem(token) for token in user_input_tokens]
    user_input_embedding = model.encode(user_input_stems, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(user_input_embedding, data_embeddings)[0]
    top_indices = np.argsort(similarities)[-4:]
    num_responses = num_of_options
    selected_indices = np.random.choice(top_indices, size=num_responses, replace=False)

    for index in selected_indices:
        answer_data.append(data[index])
    return answer_data

def choose_best_response(question, answer_data):
    #From {num_of_options} answers choose one best answer (If flag -quiz exists)
    question_embedding = model.encode([question], convert_to_tensor=True)
    answer_embeddings = model.encode(answer_data, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(question_embedding, answer_embeddings)[0]
    best_index = np.argmax(similarities)
    return best_index

#Usage

question = "-def 2 Ты бы убил всех людей на земле?"
if "-quiz" in question:
    answer = ai_poll_generator(question=question, num_of_options=int(question.split()[1]))
    best_answer_tensor = choose_best_response(question=question, answer_data=answer)
    best_answer_index = best_answer_tensor.item()
    print(answer)
    guess = int(input("Guess which answer is correct: "))
    if guess == best_answer_index:
        print("You won!")
    else:
        print(f"The correct answer is {best_answer_index}, your guess was {guess}")
elif "-def" in question:
    answer = ai_poll_generator(question=question, num_of_options=int(question.split()[1]))
    print(answer)
