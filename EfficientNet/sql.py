import pyodbc

dados_conexao = (
    "Driver={SQL Server};"
    "Server=DESKTOP-8RM56RP\SQLEXPRESS;"
    "Database=Precos;"
)

conexao = pyodbc.connect(dados_conexao)

cursor = conexao.cursor()

model_path = "EfficientNetV2S_8"
with open(model_path+'_classes', 'r') as file:
    class_names = [line.strip() for line in file]

comando = f"""SELECT Preco from Produtos
WHERE Nome = '{str(class_names[4])}'"""

cursor.execute(comando)

preco = cursor.fetchone()[0]
print("Preço:", preco)

cursor.close()
conexao.close()