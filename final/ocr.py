import worddetector, recognize_index


def ocr(path_to_image):
    indexes = worddetector.detect_words(path_to_image)
    result = []
    for index in indexes:
        result.append(recognize_index.recognize_single_index(index))
    return result

# print(ocr('../ocr1/img_15.jpg'))
