import io
from PIL import Image
from io import BytesIO
import base64
from fastapi import FastAPI, File, UploadFile
from io import StringIO
import streamlit as st
from PIL import Image, ImageFilter
from script import predict
import time
import pickle
from evaluate import execute
from pose_parser import pose_parse

st.title("Virtual Try ON")
# ảnh có sẵn để chọn
cloth1 = Image.open('./Database/val/cloth/002337_1.jpg')
cloth2 = Image.open('./Database/val/cloth/001401_1.jpg')
cloth3 = Image.open('./Database/val/cloth/003086_1.jpg')

st.sidebar.image(cloth1, caption="002337", width=100, use_column_width=False)
st.sidebar.image(cloth2, caption="002599", width=100, use_column_width=False)
st.sidebar.image(cloth3, caption="003086", width=100, use_column_width=False)


# Tải ảnh người dùng lên (Chỗ này cần sửa này Cường ơi vì đang tải ảnh nên qua thằng treamlit)
uploaded_person = st.file_uploader(
    "Upload a Photo", type=["png", "jpg", "jpeg"])
print("------uploaded_person--------", uploaded_person)
user_input = st.text_input("Enter the User Name... eg sourav")
selected = st.selectbox('Select the Item Id:', [
                        '', '002337', '002599', '003086'], format_func=lambda x: 'Select an option' if x == '' else x)


if uploaded_person is not None and user_input is not '' and selected is not '':
    with open('data.pkl', 'wb') as file:
        pickle.dump(selected, file)
    person = Image.open(uploaded_person)
    st.image(person, caption=user_input, width=100, use_column_width=False)
    st.write("Saving Image")
    bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.09)
        bar.progress(percent_complete + 1)
    person.save("./Database/val/person/"+user_input+".jpg")
    progress_bar = st.progress(0)
    st.write("Generating Mask and Pose Pairs")
    # Mấy phần trên lưu ảnh thôi, 2 dòng dưới đây xử lí, tạo pose với human parsing
    pose_parse(user_input)
    execute()
    for percent_complete in range(100):
        time.sleep(0.05)
        progress_bar.progress(percent_complete + 1)
    st.write("Please click the Click Button after Pose pairs and Masks are generated")


with open('data.pkl', 'rb') as file:
    your_variable = pickle.load(file)
    print("your_variable", your_variable)


def convertBase64():
    with open("./output/second/TOM/val/" + your_variable + "_1.jpg", 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
    print("encoded_string", encoded_string)
    return encoded_string


if st.button('Execute'):
    f = open("./Database/val_pairs.txt", "w")
    f.write(user_input+".jpg "+your_variable+"_1.jpg")
    f.close()
    # Chỗ này chạy model này, còn lại ở dưới chỉ là chỉnh output đầu ra thôi
    predict()
    from PIL import Image
    im = Image.open("./output/second/TOM/val/" + your_variable + "_1.jpg")
    width, height = im.size

# # Setting the points for cropped image
    left = width / 3
    top = 2 * height / 3
    right = 2 * width / 3
    bottom = height

# # Cropped image of above dimension
# # (It will not change orginal image)
    im1 = im.crop((left, top, right, bottom))
    newsize = (256, 256)
    im1 = im1.resize(newsize)

    # Sử dụng bộ lọc sắc nét
    im1 = im1.filter(ImageFilter.SHARPEN)
# # Shows the image in image viewer
    im1.save("./output/second/TOM/val/" + your_variable + "_1.jpg")
    execute_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.08)
        execute_bar.progress(percent_complete + 1)
    result = Image.open("./output/second/TOM/val/" + your_variable + "_1.jpg")

    st.image(result, caption="Result", width=200, use_column_width=False)

    base64_string = convertBase64()
    print("base64_string", base64_string)
    decoded_string = io.BytesIO(base64.b64decode(base64_string))
    img = Image.open(decoded_string)
    img.show()

# streamlit run app.py
