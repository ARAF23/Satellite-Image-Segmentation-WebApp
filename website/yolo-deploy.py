import gradio as gr
import PIL.Image as Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Load all models
models = {
    "YOLOv7": YOLO("v7.pt"),
    "YOLOv8": YOLO("v8.pt"),
    "YOLOv9": YOLO("v9.pt")
}

def predict_and_analyze_two(model_name, img1, img2, conf_threshold, iou_threshold, show_labels, show_boxes):
    """Predict objects in two images and generate comparative charts for counts and areas."""
    # Select the model
    model = models[model_name]

    # Process predictions for both images
    counts1, areas1, processed_img1 = process_image(model, img1, conf_threshold, iou_threshold, show_labels, show_boxes)
    counts2, areas2, processed_img2 = process_image(model, img2, conf_threshold, iou_threshold, show_labels, show_boxes)

    # Generate comparative count chart
    count_chart_path = generate_comparative_chart(counts1, counts2, img1, img2, "Object Counts")

    # Generate comparative area chart
    area_chart_path = generate_comparative_chart(areas1, areas2, img1, img2, "Total Area (pixels)")

    return processed_img1, processed_img2, count_chart_path, area_chart_path

def process_image(model, img, conf_threshold, iou_threshold, show_labels, show_boxes):
    """Process a single image with the model and return class counts, areas, and processed image."""
    results = model.predict(source=img, conf=conf_threshold, iou=iou_threshold, imgsz=640)
    class_counts = {}
    class_areas = {}
    processed_image = None

    for result in results:
        for box in result.boxes:
            # Extract class name and bbox dimensions
            cls_name = result.names[int(box.cls[0])]
            width = box.xyxy[0][2] - box.xyxy[0][0]
            height = box.xyxy[0][3] - box.xyxy[0][1]
            area = width * height

            # Update counts and areas
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            class_areas[cls_name] = class_areas.get(cls_name, 0) + area

        # Prepare the output image
        im_array = result.plot(labels=show_labels, boxes=show_boxes)  # Conditional rendering
        processed_image = Image.fromarray(im_array[..., ::-1])  # Convert to PIL.Image

    return class_counts, class_areas, processed_image

def generate_comparative_chart(data1, data2, img1, img2, chart_title):
    """Generate a comparative bar chart for two sets of data."""
    # Extract filenames for labeling
    img1_name = os.path.basename(img1) if isinstance(img1, str) else "New"
    img2_name = os.path.basename(img2) if isinstance(img2, str) else "Old"

    # Get all unique class names
    all_classes = set(data1.keys()).union(set(data2.keys()))

    # Prepare data for the chart
    values1 = [data1.get(cls, 0) for cls in all_classes]
    values2 = [data2.get(cls, 0) for cls in all_classes]

    # Generate the chart
    x = range(len(all_classes))
    plt.figure(figsize=(10, 6))
    plt.bar(x, values1, width=0.4, label=f"{img1_name}", align='center')
    plt.bar([p + 0.4 for p in x], values2, width=0.4, label=f"{img2_name}", align='center')
    plt.xticks([p + 0.2 for p in x], all_classes, rotation=45)
    plt.xlabel("Class Name")
    plt.ylabel(chart_title)
    plt.title(f"Comparative {chart_title} Between Two Images")
    plt.legend()

    # Add values on top of bars
    for i, (val1, val2) in enumerate(zip(values1, values2)):
        plt.text(i, val1, f"{val1:.1f}", ha='center', va='bottom')
        plt.text(i + 0.4, val2, f"{val2:.1f}", ha='center', va='bottom')

    # Save chart
    chart_path = f"comparative_{chart_title.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("## YOLO Model Deployment with Comparative Analysis")

    # Top controls
    with gr.Row():
        model_selection = gr.Dropdown(choices=["YOLOv7", "YOLOv8", "YOLOv9"], label="Select Model")
        confidence = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
        iou = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
        show_labels = gr.Checkbox(value=True, label="Show Labels")
        show_boxes = gr.Checkbox(value=True, label="Show Boxes")

    # First row: Input images
    with gr.Row():
        image_input1 = gr.Image(type="pil", label="New Image")
        image_input2 = gr.Image(type="pil", label="Old Image")

    # Second row: Processed images
    with gr.Row():
        output_image1 = gr.Image(type="pil", label="Prediction on New Image")
        output_image2 = gr.Image(type="pil", label="Prediction on Old Image")

    # Third row: Comparative charts
    with gr.Row():
        count_chart_output = gr.Image(label="Object Count Per Class")
        area_chart_output = gr.Image(label="Total Area Per Class")

    # Predict button
    predict_button = gr.Button("Predict and Compare")

    # Button click action
    predict_button.click(
        predict_and_analyze_two,
        inputs=[model_selection, image_input1, image_input2, confidence, iou, show_labels, show_boxes],
        outputs=[output_image1, output_image2, count_chart_output, area_chart_output],
    )

if __name__ == "__main__":
    app.launch()
