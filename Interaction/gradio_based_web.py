# Created in 2025 by Gandecheng
# create a webpage with gradio for interaction

import gradio as gr  


def update_image_input(choice):  
    if choice == "FR":  
        return {  
            ref_input: gr.update(visible=True),  
        }  
    else:  
        return {  
            ref_input: gr.update(visible=False),   
        }  
    
def test(img):
    print(img)
    return 'successs'

with gr.Blocks() as demo:  
    # title
    title = gr.Markdown(value="**Intent_based Network(IBN) for Wireless Network Access(WNA)**")
    subtitle=gr.Markdown(value="""To do""")

    
    # choose intent_router
    intent_router_radio = gr.Radio(  
        choices=[
            "Qwen2-VL-7B", 
            "Qwen2-1.5B-Instruct", 
            "Qwen2-14B-Instruct", 
            "Deepseek-R1-Distill-Qwen-7B",
            "Deepseek-R1-Distill-Qwen-14B",
        ],   
        label="Select your intent_router"  
    )  
    
    # choose WNA to simulate & optimize
    WNA_radio = gr.Radio(  
        choices=[
            "NB-IoT", 
        ],   
        label="Select your WNA to simulate & optimize"  
    )      

    with gr.Row():
        # 默认公开的失真图像输入框
        img_input = gr.Image(type="pil", 
            label="Upload the image that you want to score here",   
            visible=True
        )  

        # 默认隐藏的参考图像输入框 
        ref_input = gr.Image(type="pil", 
            label="Upload your reference image here",   
            visible=False  
        )  

    # 用户交互部分
    # 模式切换
    intent_router_radio.change(  
        fn=update_intent_router,  
        inputs=[intent_router_radio],  
        outputs=[ref_input]  
    )  

    #确认输入
    submit_button = gr.Button(value='submit') 

    with gr.Row():
    #确认输出
        grade = gr.Textbox(label="Grade / 等级")
        scores = gr.Textbox(label='Score / 评分')

    submit_button.click(  
        fn=score,  
        inputs=[img_input,mode_radio,gear,ref_input],  
        outputs=[grade,scores]  
    )  