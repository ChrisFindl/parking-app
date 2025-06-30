from detect_parking import main

# INTP/A/2025 Summer Group 1

if __name__ == "__main__":
    # Example parameters
    image_path = "res/parking1.jpg"
    total_capacity = 21
    threshold = 0.5
    save_output = True
    output_path = "detection_output.png"

    # Run detection
    result = main(
        image_path,
        total_capacity,
        threshold=threshold,
        save_output=save_output,
        output_path=output_path
    )

    # Print summary JSON
    print(result)
