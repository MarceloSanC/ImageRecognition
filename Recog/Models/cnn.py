import os, sys, datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mytools.cnn_functions import cnn_model, process_test_image

start_time = datetime.datetime.now()

train_directory_path = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Train"
test_directory_path = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Dev"

model = cnn_model(train_directory_path, test_directory_path)

test_image = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Test\\0351.jpg"
process_test_image(test_image)

end_time = datetime.datetime.now()
formatted_time = str(end_time - start_time).split(".")[0]
print("Tempo de execução:", formatted_time, '\n\n')
