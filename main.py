import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    print("=" * 60)
    print(" SISTEMA DE CLASSIFICAÇÃO TAXONÔMICA DE PINGUINS ")
    print(" Arquipélago Palmer (Antártida)")
    print("=" * 60)

    # 1. Carregamento dos dados brutos
    raw_csv_path = "penguins_lter.csv"
    print(f"\n[1] Lendo base bruta de '{raw_csv_path}'...")
    df_raw = pd.read_csv(raw_csv_path)

    # 2. Seleção e mapeamento das colunas conforme Palmer Archipelago (Antarctica) Penguin Data.txt
    # Dicionário de Dados:
    # - species: Espécie do pinguim (Adelie, Chinstrap ou Gentoo)
    # - island: Ilha onde o pinguim foi observado (Biscoe, Dream ou Torgersen)
    # - bill_length_mm: Comprimento do bico (culmen)
    # - bill_depth_mm: Profundidade/espessura do bico
    # - flipper_length_mm: Comprimento da nadadeira
    # - body_mass_g: Massa corporal
    # - sex: Sexo biológico (male ou female)

    print("\n[2] Processando e padronizando variáveis conforme especificado no .txt...")
    
    # Extrair apenas o primeiro nome da espécie (Adelie, Chinstrap, Gentoo)
    species_clean = df_raw['Species'].str.split().str[0]

    # Padronizar sexo (male/female, tratando valores nulos e ruídos como '.')
    sex_clean = df_raw['Sex'].astype(str).str.strip().str.lower()
    sex_clean = sex_clean.replace({'.': np.nan, 'nan': np.nan, '': np.nan})

    df_base = pd.DataFrame({
        'species': species_clean,
        'island': df_raw['Island'],
        'bill_length_mm': df_raw['Culmen Length (mm)'],
        'bill_depth_mm': df_raw['Culmen Depth (mm)'],
        'flipper_length_mm': df_raw['Flipper Length (mm)'],
        'body_mass_g': df_raw['Body Mass (g)'],
        'sex': sex_clean
    })

    # Salvar a base final limpa/estruturada
    output_base_path = "penguins_base.csv"
    df_base.to_csv(output_base_path, index=False)
    print(f"-> Base criada e salva com sucesso em: '{output_base_path}'")
    print(f"-> Total de registros na base: {len(df_base)}")
    print("\nVisualização das primeiras linhas da base:")
    print(df_base.head())

    print("\nDistribuição por espécie:")
    print(df_base['species'].value_counts())

    # 3. Modelagem de Machine Learning
    # Como solicitado no txt:
    # "aplica modelos de aprendizado de máquina supervisionado (como Random Forest, 
    # K-Nearest Neighbors ou Árvores de Decisão) para prever a qual espécie o animal pertence"
    print("\n" + "=" * 60)
    print("[3] Treinando Modelo de Classificação Supervisionada (Random Forest)")
    print("=" * 60)

    # Remover linhas com valores ausentes nas medições
    df_model = df_base.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']).copy()
    
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = df_model[features]
    y = df_model['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAcurácia no conjunto de teste: {acc * 100:.2f}%\n")
    print("Relatório de Classificação por Espécie:")
    print(classification_report(y_test, y_pred))

    print("Importância das características físicas na identificação:")
    for feat, imp in zip(features, clf.feature_importances_):
        print(f" - {feat:20s}: {imp * 100:.2f}%")

    # 4. Demonstração prática de predição
    print("\n" + "=" * 60)
    print("[4] Demonstração do Assistente Taxonômico em Amostras de Teste")
    print("=" * 60)
    sample_examples = pd.DataFrame([
        {'bill_length_mm': 39.1, 'bill_depth_mm': 18.7, 'flipper_length_mm': 181.0, 'body_mass_g': 3750.0},
        {'bill_length_mm': 49.5, 'bill_depth_mm': 19.0, 'flipper_length_mm': 200.0, 'body_mass_g': 3800.0},
        {'bill_length_mm': 46.5, 'bill_depth_mm': 14.8, 'flipper_length_mm': 217.0, 'body_mass_g': 5200.0}
    ])
    predictions = clf.predict(sample_examples)
    probs = clf.predict_proba(sample_examples)

    for i, (_, row) in enumerate(sample_examples.iterrows()):
        species_predicted = predictions[i]
        confidence = max(probs[i]) * 100
        print(f"\nPinguim #{i+1}:")
        print(f"  Medições: Bico={row['bill_length_mm']}x{row['bill_depth_mm']}mm | Nadadeira={row['flipper_length_mm']}mm | Peso={row['body_mass_g']}g")
        print(f"  => Espécie Prevista: {species_predicted} (Confiança: {confidence:.1f}%)")

    print("\n" + "=" * 60)
    print(" Processo concluído com sucesso! ")
    print("=" * 60)

if __name__ == "__main__":
    main()
