import pandas as pd
import mysql.connector

# Parametry połączenia
host = 'localhost'
port = '3306'
user = 'your_username'
password = 'your_password'
database = 'your_database_name'

# Połączenie się z bazą danych MySQL
try:
    conn = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    print("Connected to MySQL database successfully")
except mysql.connector.Error as err:
    print("Error: ", err)
    conn = None

# Jeśli udało się połączyć, wykonaj zapytanie i załaduj wyniki do ramki danych Pandas
if conn is not None:
    try:
        # Przykładowe zapytanie do wykonania
        query = "SELECT * FROM your_table_name"

        # Wykonanie zapytania i załadowanie wyników do ramki danych Pandas
        df = pd.read_sql(query, conn)

        # Wyświetlenie pierwszych kilku wierszy ramki danych
        print(df.head())

    except mysql.connector.Error as err:
        print("Error: ", err)

    finally:
        # Zamknięcie połączenia
        conn.close()
        print("MySQL connection is closed")
else:
    print("Unable to connect to MySQL database")
