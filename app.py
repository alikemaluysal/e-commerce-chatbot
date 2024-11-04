import streamlit as st
import json
import os
import boto3
from botocore.exceptions import ClientError

def recommend_product(user_input, products):
    bedrock = boto3.Session(profile_name="GenAI_S3_Bedrock-688567304049").client(
        service_name='bedrock-runtime',
        region_name='ap-south-1'
    )

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    prompt = f"""
    Sen bir e-ticaret ürün öneri asistanısın ve kullanıcıya pazarlık yaparak yardımcı olan profesyonel bir esnaf gibi davranıyorsun. Kullanıcıya nazik ve seviyeli bir şekilde, gereksiz samimiyetten kaçınarak ürün önerileri sunuyorsun. Kullanıcı fiyat konusunda endişeli ise, arka planda indirim kuponlarını uygula ve ona özel bir fiyat veriyormuş gibi göster.

    Yalnızca gerçek kullanıcı mesajına yanıt ver, başka hiçbir mesaj oluşturma. Kullanıcının ürün talebine veya sorusuna göre ona uygun bir yanıt ver.

    Kullanıcı Mesajı: {user_input}

    Ürün listesi:
    {json.dumps(products, ensure_ascii=False)}
    """

    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ],
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500
            }),
            contentType='application/json',
            accept='application/json'
        )

        result = json.loads(response.get("body").read())
        output_text = result.get("content", [{}])[0].get("text", "")
        return output_text
    except ClientError as err:
        st.error("Chatbot hatası: {}".format(err))
        return None

def load_products():
    if os.path.exists("products.json"):
        with open("products.json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.error("Ürün dosyası bulunamadı.")
        return []

def main():
    st.set_page_config(page_title="Ürün Öneri Chatbot", layout="centered")

    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Ürün Öneri Chatbot Uygulaması</h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    products = load_products()

    chat_container = st.container()
    with chat_container:
        st.markdown("<div style='text-align: center;'><h2>Sohbet</h2></div>", unsafe_allow_html=True)
        for message in st.session_state["messages"]:
            if message["role"] == "user":
                st.markdown(
                    f"<div style='background-color:#005f73; color: white; padding:10px; border-radius:10px; margin-bottom:5px; max-width:80%; text-align: left; float: right; clear: both;'><strong>Sen:</strong> {message['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='background-color:#0a9396; color: white; padding:10px; border-radius:10px; margin-bottom:5px; max-width:80%; text-align: left; float: left; clear: both;'><strong>Claude:</strong> {message['content']}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<hr>", unsafe_allow_html=True)
    user_input = st.text_input("Mesajınızı yazın ve Enter'a basın:", key="user_input")

    if st.button("Gönder"):
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            
            bot_response = recommend_product(user_input, products)
            if bot_response:
                st.session_state["messages"].append({"role": "bot", "content": bot_response})
            
            st.session_state["user_input"] = st.empty()

if __name__ == "__main__":
    main()
