

def conta_righe(file:list) -> int:
    return len(file)

def conta_parole(file: list) -> int:
    count_parole = 0
    for riga in file:
        parole = riga.split()
        count_parole += len(parole) 
    return count_parole

def top_frequenze(file:list) -> list:
    top_frequenze = {}
    tutte_parole = []
    for riga in file:
        tutte_parole.extend(riga.split())
    for parole in tutte_parole:
        top_frequenze[parole] = (tutte_parole.count(parole))
    ordinate = sorted(top_frequenze.items(), key=lambda x: x[1], reverse=True)

    return ordinate[:5]


def leggi_file() :
    try:
        with open("18-08\es1\input.txt","r", encoding="utf-8") as file:
            righe = file.readlines()
            print(f"Numero di righe: {conta_righe(righe)}")
            print(f"Numero di parole: {conta_parole(righe)}")
            print(f"Top 5 parole più frequenti: {top_frequenze(righe)}")
    except: 
        raise FileNotFoundError("Impossibile leggere il file.")
    
if __name__ == "__main__":
    leggi_file()