import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO

# Load the model
model = YOLO("v7.pt")

def predict_and_show(img, conf_threshold, iou_threshold, show_labels, show_boxes):
    """Predict objects in an image and return the processed image."""
    results = model.predict(source=img, conf=conf_threshold, iou=iou_threshold, imgsz=640)
    processed_image = None

    for result in results:
        # Prepare the output image
        im_array = result.plot(labels=show_labels, boxes=show_boxes)  # Conditional rendering
        processed_image = Image.fromarray(im_array[..., ::-1])  # Convert to PIL.Image

    return processed_image

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("## YOLO Model Deployment with Detection")

    # Top controls
    with gr.Row():
        confidence = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
        iou = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
        show_labels = gr.Checkbox(value=True, label="Show Labels")
        show_boxes = gr.Checkbox(value=True, label="Show Boxes")

    # Input image
    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image")

    # Output processed image
    with gr.Row():
        output_image = gr.Image(type="pil", label="Detection Result")

    # Predict button
    predict_button = gr.Button("Predict")

    # Button click action
    predict_button.click(
        predict_and_show,
        inputs=[image_input, confidence, iou, show_labels, show_boxes],
        outputs=output_image,
    )

if __name__ == "__main__":
    app.launch()
