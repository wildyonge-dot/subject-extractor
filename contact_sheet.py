import os

from PIL import Image


def create_contact_sheet(final_data, output_dir, output_filename="contact_sheet.png"):
    if not final_data:
        return None

    print("Generating contact sheet...")
    count = len(final_data)
    columns = max(1, int(count ** 0.5))
    rows = (count + columns - 1) // columns

    thumb_size = 200
    padding = 20
    sheet_width = columns * thumb_size + (columns + 1) * padding
    sheet_height = rows * thumb_size + (rows + 1) * padding
    sheet = Image.new("RGBA", (sheet_width, sheet_height), (30, 30, 30, 255))

    for index, item in enumerate(final_data):
        row = index // columns
        column = index % columns
        x = padding + column * (thumb_size + padding)
        y = padding + row * (thumb_size + padding)
        image_path = os.path.join(output_dir, item["filename"])

        try:
            image = Image.open(image_path).convert("RGBA")
            image.thumbnail((thumb_size, thumb_size))
            offset_x = x + (thumb_size - image.width) // 2
            offset_y = y + (thumb_size - image.height) // 2
            background = Image.new("RGBA", image.size, (100, 100, 100, 255))
            sheet.paste(background, (offset_x, offset_y))
            sheet.paste(image, (offset_x, offset_y), mask=image)
        except Exception as exc:
            print(f"Error adding {item['filename']} to contact sheet: {exc}")

    output_path = os.path.join(output_dir, output_filename)
    sheet.save(output_path)
    return output_path
