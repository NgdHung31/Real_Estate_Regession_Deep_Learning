import gradio as gr
from inference import inference_model

############################### PRICE PREDICTION ###############################
################################################################################
# Hàm dự đoán giá
def predict_price(property_type, area, floors, bedrooms, toilets, legal_status, furniture, project_name, district, distance):
    price = inference_model(property_type, area, floors, bedrooms, toilets, legal_status, furniture, project_name, district, distance)
    return f"{price:,.2f} tỷ đồng"

# Giao diện trang chủ
def home():
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Markdown("""
            <div style="text-align: center;">
                <h1>Chào mừng đến với ứng dụng Hệ thống Bất động sản $mart 🏠!</h1>
                <p>Chúng tôi cung cấp công cụ dự đoán giá bất động sản chính xác dựa trên các tham số đầu vào và tư vấn qua chatbot.</p>
            </div>
            """)
            gr.Image("assets/pexels-binyaminmellish-106399.jpg", elem_id="real-estate-image", show_label=False, interactive=False)
        
            gr.Markdown("""
            <div style="text-align: center;">
                <h2>Tại sao lại chọn công cụ dự đoán của chúng tôi?</h2>
                <ul style="list-style-type:none; text-align: left; display: inline-block;">
                    <li>✔️ Dự đoán chính xác dựa trên dữ liệu thực tế</li>
                    <li>✔️ Giao diện dễ sử dụng</li>
                    <li>✔️ Nhanh chóng và đáng tin cậy</li>
                </ul>
                <h2>Cách sử dụng giao diện dự đoán</h2>
                <p>Thực hiện theo các bước đơn giản sau để nhận dự đoán giá bất động sản của bạn:</p>
                <ol style="text-align: left; display: inline-block;">
                    <li>Nhấp vào tab 'Dự đoán giá'</li>
                    <li>Nhập thông tin chi tiết về các thuộc tính bất động sản bạn cần hỏi đáp</li>
                    <li>Nhấp vào nút 'Dự đoán'</li>
                    <li>Nhận giá dự đoán ngay lập tức</li>
                </ol>
                <h2>Hỗ trợ chatbot</h2>
                <p>✔️ Nếu bạn có bất kỳ câu hỏi nào hoặc cần hỗ trợ thêm, bạn có thể sử dụng chatbot của chúng tôi để được hỗ trợ và tư vấn ngay lập tức.</p>
                <h2>Cách sử dụng chatbot</h2>
                <p>Thực hiện theo các bước đơn giản sau để sử dụng chatbot:</p>
                <ol style="text-align: left; display: inline-block;">
                    <li>Nhấp vào tab 'Chatbot'</li>
                    <li>Nhập thông tin bạn cần hỏi</li>
                    <li>Nhận thông tin từ chatbot</li>
                    <li>Sau đó bạn có thể trò chuyện với chatbot</li>
                </ol>               
                <h2>Về chúng tôi</h2>
                <p>Nhóm chúng tôi bao gồm các sinh viên đến từ Trường Đại học Khoa học Tự Nhiên - Đại học Quốc gia TP.HCM.</p>
                <p>Đây là sản phẩm thuộc học phần Đồ án tốt nghiệp do nhóm chúng tôi nghiên cứu và sản xuất.</p>
                <h2>Liên hệ với chúng tôi</h2>
                <p>Nếu bạn có bất kỳ câu hỏi nào hoặc cần hỗ trợ thêm, đừng ngần ngại liên hệ với chúng tôi thông qua email này: <a href="mailto:20280047@student.hcmus.edu.vn"></a>.</p>
            </div>
            """)
    return demo

# Giao diện dự đoán
def predict_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
        <style>
            body {font-family: Arial, sans-serif;}
            h1 {color: #4CAF50;}
            .gradio-container {background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);}
            .gradio-row {margin-bottom: 10px;}
        </style>
        """)
        gr.Markdown("<h1 style='text-align: center;'>💰 Real Estate Price Prediction</h1>")
        gr.Image("assets/pexels-binyaminmellish-1396122.jpg", elem_id="real-estate-image", show_label=False, interactive=False)

        with gr.Row():
            with gr.Column():
                property_type_input = gr.Dropdown(
                    label="🏠 Loại hình bất động sản",
                    choices=["Chung cư", "Nhà ngõ, hẻm", "Nhà mặt phố", "Nhà biệt thự", "Nhà phố liền kề"],
                    value="Chung cư"
                )
                district_input = gr.Dropdown(
                    label="📍 Quận/Huyện",
                    choices=["Quận 1", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7", "Quận 8", "Quận 10", "Quận 11", "Quận 12", "Bình Chánh", "Bình Tân", "Bình Thạnh", "Cần Giờ", "Củ Chi", "Hóc Môn", "Nhà Bè", "Phú Nhuận", "Gò Vấp", "Tân Bình", "Tân Phú", "TP Thủ Đức"],
                    value="TP Thủ Đức"
                )
                distance_input = gr.Slider(
                    label="🛣️ Khoảng cách đến trung tâm thành phố (chợ Bến Thành) (km)",
                    minimum=0.1, maximum=200, step=0.1, value=50
                )
                area_input = gr.Slider(
                    label="📏 Diện tích (m²)",
                    minimum=7, maximum=170, step=0.1, value=50
                )
                legal_status_input = gr.Dropdown(
                    label="📜 Tình trạng pháp lý",
                    choices=["Đã có sổ", "Giấy tờ khác", "Đang chờ sổ", "Unknown"],
                    value="Đã có sổ"
                )
            with gr.Column():
                project_name_input = gr.Textbox(
                    label="📝 Tên dự án",
                    value="Unknown"
                )
                furniture_input = gr.Dropdown(
                    label="🛋️ Nội thất",
                    choices=["Không có nội thất", "Có nội thất"],
                    value="Không có nội thất"
                )
                floors_input = gr.Slider(
                    label="🏢 Số tầng",
                    minimum=1, maximum=30, step=1, value=1
                )
                bedrooms_input = gr.Slider(
                    label="🛏️ Số phòng ngủ",
                    minimum=1, maximum=60, step=1, value=1
                )
                toilets_input = gr.Slider(
                    label="🚽 Số toilets",
                    minimum=1, maximum=7, step=1, value=1
                )
            
        predict_btn = gr.Button("Dự đoán", variant="primary")
        result = gr.Textbox(label="💰 Giá dự đoán")
        
        predict_btn.click(predict_price, 
                          inputs=[property_type_input, area_input,
                                  floors_input, bedrooms_input, toilets_input,
                                  legal_status_input, furniture_input, project_name_input, 
                                  district_input, distance_input],
                          outputs=result)
    
    return demo


############################### CHATBOT ###############################
#######################################################################
import gradio as gr
import os
import codecs
import openai
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
#from dotenv import load_dotenv

#load_dotenv()

pc = Pinecone(api_key="os.getenv()")
cloud = os.environ.get("PINECONE_CLOUD") or "aws"
region = os.environ.get("PINECONE_REGION") or "us-east-1"

spec = ServerlessSpec(cloud=cloud, region=region)

# Set up OpenAI API
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key="os.getenv()")
# Load the Vietnamese embedding model
model = SentenceTransformer('intfloat/multilingual-e5-small')

# Function to get embeddings


def get_embeddings(sentences):
    tokenizer_sent = [tokenize(sent) for sent in sentences]
    embeddings = model.encode(tokenizer_sent)
    return embeddings


# Function to create metadata


def create_meta_batch(sentences):
    return [{"text": sent} for sent in sentences]


# Function to read sentences from a file


def read_sentences_from_file(file_path): # chunk
    with open(file_path, "r", encoding="utf-8-sig") as file:
        sentences = file.readlines()
    # Remove any leading/trailing whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def process_and_upsert(sentences, batch_size=100, index_name="rag-index"):
    print(pc.list_indexes().names())
    if index_name not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=384,  # dimensionality of text-embedding-ada-002
            metric="cosine",
            spec=spec,
        )

        # connect to index
        index = pc.Index(index_name)

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            embeddings = get_embeddings(batch)
            ids = [str(i + j) for j in range(len(batch))]
            meta_batch = create_meta_batch(batch)
            vectors = [
                {"id": id_, "values": embedding.tolist(), "metadata": meta}
                for id_, embedding, meta in zip(ids, embeddings, meta_batch)
            ]
            index.upsert(vectors)
    else:
        index = pc.Index(index_name)
    return index



# Initialize vector database
file_path = "./content/data.txt"
sentences = read_sentences_from_file(file_path)
index = process_and_upsert(sentences, batch_size=100, index_name="rag-index")

# A dictionary to store conversation history for each session
conversation_history = {}

# Define the system prompt with rules
system_prompt = """
You are a chatbot designed to act as a real estate consulting assistant. Your main role is to provide users with accurate and useful information, advice, and insights on real estate queries. When answering questions, you must adhere to the following principles:

1. Accuracy and Relevance: Ensure your answers are based on current and relevant real estate data and trends, but do not reference or refer to specific data sources in your responses.
2. Scope Management: If a query is beyond your capabilities or tools, instruct users on how to find additional help or suggest alternative methods to find the information they require.
3. Scope Management: If a query is beyond your capabilities or tools, instruct users on how to find additional help or suggest alternative methods to find the information they require.
4. Information Gathering: Ask for additional information if it is necessary to provide a more accurate answer.
5. Language Consistency: Always reply in the user's language. If the user speaks Vietnamese, you should reply in Vietnamese.

I. Identifying User Needs:
1. Ask the user what type of real estate service they are interested in (buying, selling, renting, or investing).
2. Gather basic information about their requirements (location, budget, property type, etc.).

II. Providing Information and Options
1. Based on the user’s requirements, provide information on available properties or services.
2. Share links or details of properties that match their criteria.
3. Offer to schedule viewings or consultations if applicable.

III. Answering Questions
1. Be prepared to answer common questions related to real estate transactions (e.g., mortgage rates, property taxes, neighborhood details, etc.).
2. Provide clear and concise answers.
Example:
"We have several properties that match your criteria. Here are a few options: [Property 1 Details], [Property 2 Details], [Property 3 Details]. Would you like to schedule a viewing or need more information on any of these?"

General Tips:
1. Maintain a friendly and professional tone throughout the conversation.
2. Be concise and to the point, avoiding overly technical jargon.
3. Ensure quick and accurate responses to user queries.
4. Offer personalized assistance based on user inputs.
5. Ensure user data privacy and confidentiality at all times.

You will be provided with additional documents containing information on properties. Use these documents to answer questions based on the provided data.  Distance is calculated from the location of the house to the city center.
If the document does not contain the information needed to answer a question, simply write:
"Tôi không có thông tin về vấn đề này."
"""


def first_conversational_rag(session_id):
    conversation_history[session_id] = {
        "messages": [{"role": "system", "content": system_prompt}],
        "info": {},
    }


def conversational_rag(session_id, question, history):
    limit = 3750
    # res = openai.Embedding.create(
    #     input=[question],
    #     engine="text-embedding-ada-002"
    # )
    # xq = res['data'][0]['embedding']
    tokenizer_sent = [tokenize(question)]
    xq = model.encode(tokenizer_sent)
    xq = xq.tolist()[0]
    contexts = []
    time_waited = 0
    while len(contexts) < 3 and time_waited < 60 * 5:
        res = index.query(vector=xq, top_k=3, include_metadata=True)
        contexts = contexts + [x["metadata"]["text"] for x in res["matches"]]
        print(f"Retrieved {contexts}")
        time.sleep(2)
        time_waited += 20
    if time_waited >= 60 * 5:
        print("Timed out waiting for contexts to be retrieved.")
        contexts = ["No documents retrieved. Try to answer the question yourself!"]
    for i in range(1, len(contexts)):
        if len("\n".join(contexts[:i])) >= limit:
            prompt = "\n".join(contexts[: i - 1])
            break
        elif i == len(contexts) - 1:
            prompt = "\n".join(contexts)
    # Include stored information in the context
    stored_history = conversation_history[session_id]["messages"]
    stored_history.append(
        {"role": "system", "content": "Additional document:" + prompt}
    )
    stored_history.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=stored_history, max_tokens=350, temperature=0.7
    )
    answer = response.choices[0].message.content
    conversation_history[session_id]["messages"].append(
        {"role": "user", "content": question}
    )
    conversation_history[session_id]["messages"].append(
        {"role": "assistant", "content": answer}
    )
    history.append((question, answer))
    return "", history



def clear_history(session_id):
    if session_id in conversation_history:
        del conversation_history[session_id]
    first_conversational_rag(session_id)
    gr.Info("Xóa lịch sử trò chuyện.")
    return


def open_ui(session_id):
    if session_id not in conversation_history:
        first_conversational_rag(session_id)
        return gr.update(visible=True)
    return None, None

def chatbot_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
        <style>
            body {font-family: Arial, sans-serif;}
            h1 {color: #4CAF50;}
            .gradio-container {background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);}
            .gradio-row {margin-bottom: 10px;}
        </style>
        """)
        gr.Markdown("<h1 style='text-align: center;'>🤖 Chatbot</h1>")
        gr.Image("assets/pexels-binyaminmellish-1396132.jpg", elem_id="real-estate-image", show_label=False, interactive=False)
        
        session_id = gr.Textbox(label="Nhập vào đây tên đoạn hội thoại:")
        start_button = gr.Button("Bắt đầu cuộc hội thoại", variant="primary")

        with gr.Column(visible=False) as main_row:
            chatbot = gr.Chatbot(
                value=[
                    [
                        None,
                        "Xin chào, Chào mừng bạn đến với công ty bất động sản $mart Home. Tôi là trợ lý ảo ở đây để giúp bạn giải quyết các nhu cầu về bất động sản. Hôm nay tôi có thể hỗ trợ bạn như thế nào?",
                    ]
                ],
                label="Chatbot",
                placeholder="Chatbot is ready to answer your questions.",
            )
            question = gr.Textbox(label="Câu hỏi của bạn")
            # submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Xóa lịch sử cuộc trò chuyện")

            question.submit(
                conversational_rag,
                inputs=[session_id, question, chatbot],
                outputs=[question, chatbot],
            )
            clear_btn.click(clear_history, inputs=[session_id], outputs=[])

        start_button.click(open_ui, inputs=[session_id], outputs=[main_row])

# Main 
with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as main:
    
    with gr.Tabs():
        
        with gr.TabItem("🏠 Home"):
            home()
            
        with gr.TabItem("💰 Price Prediction"):
            predict_interface()
            
        with gr.TabItem("🤖 Chatbot"):
            chatbot_interface()

main.launch(inbrowser=True)
