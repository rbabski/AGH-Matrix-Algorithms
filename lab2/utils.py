import numpy as np

# ---------------------------Strassen-----------------------

def multiply_strassen(M1, M2, count=None):
    """
    Zoptymalizowana wersja algorytmu Strassena do mnożenia macierzy
    z redukcją zużycia pamięci i poprawą wydajności czasowej.
    """
    if count is None:
        count = {'add': 0, 'mul': 0}
    
    r1, c1 = M1.shape
    r2, c2 = M2.shape

    if c1 != r2:
        raise Exception("Invalid matrix sizes.")

    # Bazowy przypadek - użyj bezpośredniego mnożenia
    if min(r1, c1, r2, c2) <= 64:  # Zwiększony próg
        count['mul'] += r1 * c1 * c2
        count['add'] += r1 * (c1 - 1) * c2
        return M1 @ M2, count
    
    # Sprawdź czy macierze są parzyste
    if r1 % 2 == 0 and c1 % 2 == 0 and r2 % 2 == 0 and c2 % 2 == 0:
        # Oblicz punkty podziału - używaj widoków
        mid_r1, mid_c1 = r1//2, c1//2
        mid_r2, mid_c2 = r2//2, c2//2
        mid_c_res = c2//2
        
        # Widoki na podmacierze (bez kopiowania)
        A1 = M1[:mid_r1, :mid_c1]
        B1 = M1[:mid_r1, mid_c1:]
        C1 = M1[mid_r1:, :mid_c1]
        D1 = M1[mid_r1:, mid_c1:]
        
        A2 = M2[:mid_r2, :mid_c2]
        B2 = M2[:mid_r2, mid_c2:]
        C2 = M2[mid_r2:, :mid_c2]
        D2 = M2[mid_r2:, mid_c2:]
        
        # Obliczenia pośrednie Strassena z optymalizacją pamięci
        # M1 = A1 + D1, M2 = A2 + D2
        temp1 = np.empty_like(A1)
        temp2 = np.empty_like(A2)
        np.add(A1, D1, out=temp1)
        np.add(A2, D2, out=temp2)
        count['add'] += A1.size * 2
        P1, count = multiply_strassen(temp1, temp2, count)
        
        # M1 = C1 + D1, M2 = A2
        np.add(C1, D1, out=temp1)
        P2, count = multiply_strassen(temp1, A2, count)
        count['add'] += C1.size
        
        # M1 = A1, M2 = B2 - D2
        np.subtract(B2, D2, out=temp2)
        P3, count = multiply_strassen(A1, temp2, count)
        count['add'] += B2.size
        
        # M1 = D1, M2 = C2 - A2
        np.subtract(C2, A2, out=temp2)
        P4, count = multiply_strassen(D1, temp2, count)
        count['add'] += C2.size
        
        # M1 = A1 + B1, M2 = D2
        np.add(A1, B1, out=temp1)
        P5, count = multiply_strassen(temp1, D2, count)
        count['add'] += A1.size
        
        # M1 = C1 - A1, M2 = A2 + B2
        np.subtract(C1, A1, out=temp1)
        np.add(A2, B2, out=temp2)
        P6, count = multiply_strassen(temp1, temp2, count)
        count['add'] += C1.size + A2.size
        
        # M1 = B1 - D1, M2 = C2 + D2
        np.subtract(B1, D1, out=temp1)
        np.add(C2, D2, out=temp2)
        P7, count = multiply_strassen(temp1, temp2, count)
        count['add'] += B1.size + C2.size
        
        # Oblicz wynikowe bloki z ponownym wykorzystaniem pamięci
        # UL = P1 + P4 - P5 + P7
        UL = np.add(P1, P4, out=P1)    # reuse P1
        np.subtract(UL, P5, out=UL)
        np.add(UL, P7, out=UL)
        
        # UR = P3 + P5
        UR = np.add(P3, P5, out=P3)    # reuse P3
        
        # LL = P2 + P4
        LL = np.add(P2, P4, out=P2)    # reuse P2
        
        # LR = P1 - P2 + P3 + P6
        LR = np.subtract(P1, P2, out=P1)  # reuse P1 (już niepotrzebne)
        np.add(LR, P3, out=LR)
        np.add(LR, P6, out=LR)
        
        count['add'] += P1.size * 8  # 8 operacji na macierzach tego rozmiaru
        
        # Zbuduj wynik bez użycia np.block
        res = np.empty((r1, c2), dtype=M1.dtype)
        res[:mid_r1, :mid_c_res] = UL
        res[:mid_r1, mid_c_res:] = UR
        res[mid_r1:, :mid_c_res] = LL
        res[mid_r1:, mid_c_res:] = LR
        
        return res, count
    
    else:
        # Dla macierzy nieparzystych - użyj metody dzielenia na bloki
        return multiply_strassen_odd_optimized(M1, M2, count)

def multiply_strassen_odd_optimized(M1, M2, count):
    """
    Optymalizowana wersja dla macierzy o nieparzystych wymiarach.
    """
    r1, c1 = M1.shape
    r2, c2 = M2.shape
    
    # Dzielenie macierzy na bloki
    M1_block, v1, w1, s1 = M1[:-1, :-1], M1[:-1, -1:], M1[-1:, :-1], M1[-1:, -1:]
    M2_block, v2, w2, s2 = M2[:-1, :-1], M2[:-1, -1:], M2[-1:, :-1], M2[-1:, -1:]
    
    # Oblicz M1_block * M2_block
    M1M2, count = multiply_strassen(M1_block, M2_block, count)
    
    # Prealokuj pamięć dla wyników
    UL = np.empty_like(M1M2)
    UR = np.empty((M1_block.shape[0], 1), dtype=M1.dtype)
    LL = np.empty((1, M2_block.shape[1]), dtype=M1.dtype)
    
    # Obliczenia z optymalizacją pamięci
    # UL = M1M2 + v1 @ w2
    v1w2 = v1 @ w2
    np.add(M1M2, v1w2, out=UL)
    
    # UR = M1_block @ v2 + v1 * s2
    M1v2 = M1_block @ v2
    v1s2 = v1 * s2[0,0] if s2.size == 1 else v1 @ s2
    np.add(M1v2, v1s2, out=UR)
    
    # LL = w1 @ M2_block + s1 * w2
    w1M2 = w1 @ M2_block
    s1w2 = s1[0,0] * w2 if s1.size == 1 else s1 @ w2
    np.add(w1M2, s1w2, out=LL)
    
    # LR = w1 @ v2 + s1 * s2
    w1v2 = w1 @ v2
    s1s2 = s1[0,0] * s2[0,0] if s1.size == 1 and s2.size == 1 else s1 @ s2
    LR = w1v2 + s1s2
    
    # Aktualizuj licznik operacji
    count["add"] += UL.size + UR.size + LL.size + LR.size
    count["mul"] += (v1.size + M1_block.size + M2_block.size + 
                    w2.size + w1.size + 1)  # Uproszczone liczenie
    
    # Zbuduj wynikową macierz
    res = np.empty((r1, c2), dtype=M1.dtype)
    res[:-1, :-1] = UL
    res[:-1, -1:] = UR
    res[-1:, :-1] = LL
    res[-1:, -1:] = LR
    
    return res, count



#------------------------------------Binet------------------


def multiply_binet(M1, M2, count=None):
    """
    Zoptymalizowana wersja algorytmu Binet'a do mnożenia macierzy
    z redukcją zużycia pamięci i poprawą wydajności czasowej.
    """
    if count is None: 
        count = {'add': 0, 'mul': 0}

    r1, c1 = M1.shape
    r2, c2 = M2.shape

    if c1 != r2:
        raise Exception("Invalid matrix sizes.")
    
    # Bazowy przypadek - użyj bezpośredniego mnożenia
    if min(r1, c1, r2, c2) <= 32:  # Zwiększony próg
        count['mul'] += r1 * c1 * c2
        count['add'] += r1 * (c1 - 1) * c2
        return M1 @ M2, count
    
    # Oblicz punkty podziału
    mid_r1, mid_c1 = r1//2, c1//2
    mid_r2, mid_c2 = r2//2, c2//2
    mid_c_res = c2//2
    
    # Używaj widoków zamiast kopiowania podmacierzy
    A1 = M1[:mid_r1, :mid_c1]
    B1 = M1[:mid_r1, mid_c1:]
    C1 = M1[mid_r1:, :mid_c1]
    D1 = M1[mid_r1:, mid_c1:]
    
    A2 = M2[:mid_r2, :mid_c2]
    B2 = M2[:mid_r2, mid_c2:]
    C2 = M2[mid_r2:, :mid_c2]
    D2 = M2[mid_r2:, mid_c2:]
    
    # Oblicz iloczyny pośrednie
    P1, count = multiply_binet(A1, A2, count)
    P2, count = multiply_binet(B1, C2, count)
    P3, count = multiply_binet(A1, B2, count)
    P4, count = multiply_binet(B1, D2, count)
    P5, count = multiply_binet(C1, A2, count)
    P6, count = multiply_binet(D1, C2, count)
    P7, count = multiply_binet(C1, B2, count)
    P8, count = multiply_binet(D1, D2, count)
    
    # Optymalizacja dodawania - ponowne wykorzystanie pamięci
    UL = np.add(P1, P2, out=P1)  # reuse P1 memory
    UR = np.add(P3, P4, out=P3)  # reuse P3 memory  
    LL = np.add(P5, P6, out=P5)  # reuse P5 memory
    LR = np.add(P7, P8, out=P7)  # reuse P7 memory
    
    count["add"] += UL.size + UR.size + LL.size + LR.size
    
    # Zbuduj wynikową macierz bez użycia np.block
    res = np.empty((r1, c2), dtype=M1.dtype)
    res[:mid_r1, :mid_c_res] = UL
    res[:mid_r1, mid_c_res:] = UR
    res[mid_r1:, :mid_c_res] = LL
    res[mid_r1:, mid_c_res:] = LR
    
    return res, count