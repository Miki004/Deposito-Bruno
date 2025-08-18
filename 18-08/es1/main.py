

def conta_righe(file:list) -> int:
    return len(file)

def conta_parole(file: list) -> int:
    count_parole = 0
    for riga in file:
        parole = riga.split()
        count_parole += len(parole) 
    return count_parole




def leggi_file() :
    try:
        with open("18-08\es1\input.txt","r", encoding="utf-8") as file:
            righe = file.readlines()
            print(f"Numero di righe: {conta_righe(righe)}")
            print(f"Numero di parole: {conta_parole(righe)}")
    except: 
        raise FileNotFoundError("Impossibile leggere il file.")
    
if __name__ == "__main__":
    leggi_file()