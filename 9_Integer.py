import math

def generate_gamma_lut(gamma_value, entries_to_show=5):
    print(f"Menghitung {entries_to_show} entri pertama LUT dengan gamma = {gamma_value}")
    
    lut_entries = []
    
    # Iterasi untuk menghitung entri
    for k in range(entries_to_show):

        normalized_input = k / 255.0
        normalized_output = math.pow(normalized_input, 1.0 / gamma_value)
        final_value = round(255.0 * normalized_output)
        
        lut_entries.append(int(final_value))
        
    print(f"Lima entri pertama dari LUT: {lut_entries}")
    print("--------------------")

if __name__ == "__main__":
    # Contoh penggunaan dengan gamma standar 2.2
    generate_gamma_lut(gamma_value=2.2)

