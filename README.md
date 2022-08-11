## WEB MOTORS: LEAD CLASSIFICATION
![image](https://user-images.githubusercontent.com/85720162/182043755-1849f702-043a-43b5-9681-b38bab359fb5.png)


------------------------------------------------------------------
## Sobre o Projeto
Esse projeto foi realizado como teste de conhecimento em um case técnico, os dados utlizados são fictícios e foram disponibilizados pela empresa. O objetivo desse projeto é extrair insights que possam ajudar anúnciantes a anúnciarem melhore seus veículos na plataforma e construir um modelo de Machine Learning para classificar se um anúncio vai/não gerar leads.

O projeto foi realizado de forma cíclica (metodologia CRISP-DM), sendo cada iteração uma melhoria do que foi desenvolvido anteriormente.

**Esse projeto representa o primeiro ciclo da solução.**

Acesse o notebook de desenvolvimento do projeto nesse link: 
https://github.com/lbVictor/WebMotors_Classification/blob/main/WM_classification_development_v01.ipynb

## Sobre a Empresa 
Webmotors é uma plataforma de anúncios, onde compradores encontram vendedores de veículos. Não é possível comprar ou vender diretamente no site da Webmotors. Ou seja, o site é um intermediário. Assim, o interessado pode entrar em contato com quem disponibiliza o produto. Depois disso a negociação e a compra serão efetuadas fora do site. Para o comprador é uma ótima maneira de procurar seu carro ideal sem sair de casa, além de poder comparar preços. Já para o vendedor é uma oportunidade de ter uma grande alcance de público que está interessado em um automóvel igual ao dele.


## Estrutura do Projeto
### Entendimento do Negócio
00. **Problema de Negócio:** A Webmotors dispõe de diversos produtos digitais para solucionar problemas durante a compra, utilização e venda de um veículo, sendo um de seus principais e mais conhecidos o classificado. Para o classificado utilizamos o lead model para monetizar nosso produto. Um lead é uma demonstração de interesse de um comprador para um vendedor. 

    **Desafio Proposto:** A fim de melhorar o desempenho dos anúncios, determine ao menos uma alternativa para potencializar o recebimento de leads dos anúncios. Para tal, elaboramos um roteiro em duas partes para te guiar neste desafio:

      - **Parte 1:** Faça uma exploração dos dados e mostre como a quantidade de leads varia de acordo com as outras variáveis.
      
      - **Parte 2:** Proponha um modelo para determinar se um anúncio receberá lead. Que outras informações, as quais você não teve acesso, poderiam ajudar a chegar a conclusões melhores?

      - **Desafio Bônus:** Proponha um modelo para determinar a quantidade de leads que um anúncio irá receber

	**Soluções que o modelo de classificação poderia gerar:**	
	- Identificação de anúncios que não gerariam leads, para oferecer insights (pagos ou gratuitos) para os anúnciantes.
	- Identificar quais anúncios receberiam leads e calcular a receita equivalente.
	- Otimizar a plataforma (filtros, possibilidade de customização por parte dos anúnciantes, etc.)

	
  	**Features disponíveis no dataset:**	  
	
   	| Feature                                    | Descrição                                         |
   	| ------------------------------------------ | ------------------------------------------------- |
   	| cod_anuncio                                |                                 código do anúncio |                    
    | cod_cliente                                |                              código do anunciante |          
    | cod_tipo_pessoa                            |                    tipo de anunciante: PF=1, PJ=2 |              
    | prioridade                                 |  prioridade do anúncio (1=alta, 2-média, 3-baixa) |         
    | views                                      |            quantidade de visualizações no anúncio |    
    | cliques_telefone*                          |       quantidade de cliques no telefone anunciado |                
    | cod_marca_veiculo                          |                        código da marca do veículo |                
    | cod_modelo_veiculo                         |                          código do modelo veículo |                 
    | cod_versao_veiculo                         |                       código da versão do veículo |                 
    | ano_modelo                                 |                             ano-modelo do veículo |         
    | cep_2dig                                   |       dois primeiros dígitos do cep do anunciante |       
    | uf_cidade                                  |                         UF e cidade do anunciante |        
    | vlr_anuncio                                |                       valor do veículo no anúncio |          
    | qtd_fotos                                  |                    quantidade de fotos no anúncio |        
    | km_veiculo                                 |                           Kilometragem do veículo |         
    | vlr_mercado                                |           valor de referência do veículo na praça |          
    | flg_unico_dono                             |                           indicador de único dono |             
    | flg_licenciado                             |                 indicador de licenciamento em dia |             
    | flg_ipva_pago                              |                          indicador de IPVA em dia |            
    | flg_todas_revisoes_concessionaria          | indicador realização de todas as revisões na c... |              
    | flg_todas_revisoes_agenda_veiculo          | indicador realização de todas as revisões prev... |                    
    | flg_garantia_fabrica                       |       indicador de veículo em garantia de fábrica |                   
    | flg_blindado                               |                            indicador de blindagem |           
    | flg_aceita_troca                           |        indicador de que o anunciante aceita troca |               
    | flg_adaptado_pcd                           | indicador de veículo adaptado para pessoa com ... |               
    | combustivel                                |                      especificação de combustível |          
    | cambio                                     |                           especificação de câmbio |     
    | portas                                     |                                  número de portas |     
    | alarme                                     |                                presença de alarme |     
    | airbag                                     |                                presença de airbag |     
    | arquente                                   |                             presença de ar quente |       
    | bancocouro                                 |                        presença de banco de couro |         
    | arcondic                                   |                       presença de ar condicionado |       
    | abs                                        |                             presença de freio abs |  
    | desembtras                                 |                 presença de desembaçador traseiro |         
    | travaeletr                                 |                      presença de travas elétricas |         
    | vidroseletr                                |                      presença de vidros elétricos |          
    | rodasliga                                  |                    presença de rodas de liga-leve |        
    | sensorchuva                                |                       presença de sensor de chuva |          
    | sensorestacion                             |              presença de sensor de estacionamento |   
    | leads                                      |                       tota de propostas recebidas |
   
		
### Entendimento e Preparação dos Dados

01. **Data Cleaning & Data Description:** Nessa etapa, foi realizado a divisão do conjunto de dados entre Treino e Teste; analisado a descrição das features; verificado o tamanho do conjunto de dados; verificado dados duplicados; verificado valores faltantes (devido ao tempo disponibilizado para esse primeiro ciclo, os valores faltantes do dataset foram removidos); verificado e alterado o tipo dos dados; realizado uma análise estatística descritiva para avaliar os dados categóricos e númericos e definir ações necessárias para limpeza ou análises posteriores.

02. **Feature Engineering:** Nessa etapa, foi desenvolvido um Mapa Mental para relacionar as caracteríticas do fenômeno de geração de leads pela Webmotors, com o objetivo de: identificar features interessantes para serem analisadas; criar hipóteses de negócio que possam vir a gerar insights acionáveis; conduzir a EDA; facilitar o cruzamento de informações para realizar o Feature Engineering.

	![image](https://user-images.githubusercontent.com/85720162/184257365-9d6ac4af-cc86-4597-b918-e60845543a37.png)

         
      - Hipóteses geradas no primeiro ciclo: 26
      - Hipóteses priorizadas para validação: 9
      
      Features não presentes no conjunto de dados que podem ser úteis individualmente ou na geração de outras features para gerar melhores resultados nas análises e modelagem: 
        
          01. Cor do veículo.
          02. Qualidade das fotos.
          03. Local das fotos (interna/externa, rodas/pneus, capô, motor, ect.).
          04. Quantidade de palavras nas descrições dos anúncios.
          05. Avaliação dos anúnciantes (quando houver).
          06. Gênero do cliente.
          07. Tempo média dos leads na página do anúncio.
          08. Idade do cliente.
          09. Média da quantidade de anúncios que o cliente visitou antes de se interessar pelo anúncio.
          10. Dia do mês.
          11. Mês.
          12. Horário que o anúncio recebeu cada lead.
          13. Tempo que o anúncio esta em aberto.
          
      Features criadas no primeiro ciclo: 
      
          01. Dividido a feature uf_cidade em duas.
          02. Variável binária informando se o anúncio recebeu ou não recebeu leads.
          03. Quantidade média de leads por estado.
          04. Quantidade de itens opcionais de fábrica por veículo anunciado.
          05. Variável binária informando se o IPVA foi pago E o Licenciamento está em dia.
          06. Variável binária informando se o valor do anúncio é inferior/igual ao valor de mercado, ou se é superior.
          
03. **Data Filtering:** Nessa etapa, foi excluído a feature uf_cidade utilizada no split para geração das variáveis separadas e filtrado os dados que continham a variável porta igual a 0 e a variável views = 0.

04. **Exploratory Data Analysis (EDA):** Nessa etapa, foi realizado a análise de algumas features (devido ao tempo, somente algumas features foram analisadas, o critério de seleção foi baseado na possibilidade do anunciante poder alterar/adaptar algo no anúncio e aumentar as chances de receber leads). Também foi validado as 9 hipóteses de negócio geradas na etapa do Feature Engineering.
	
	Insight gerados nas análises das Features com base apenas nos dados do dataset:
		
	- A quantidade ideal de fotos em um anúncio é de 08 fotos, representando 55% dos anúncios que tiveram leads gerados.
	- Veículos licenciados tem maiores chances de receberem leads. De todos os veículos licenciados do dataset 80.93% geraram leads.
	- Veículos com IPVA pago tem maiores chances de receberem leads. De todos os veículos com IPVA pago do dataset 79.90% geraram leads.
	- Anunciantes que aceitam troca tem maiores chances de receberem leads. De todos os anúncios que aceitam troca, 83.49% geraram leads.

	Curiosidades: 
		
	- Não existe uma diferença significativa de leads com base na prioridade do anúncio.
	- Os valores dos anúncios serem maiores ou menores que os valores do mercado não impactam no recebimento de leads.

	Insights retirados da validaçãod e hipóteses:
	
	- Hipótese 02 [Falsa] - Duas marcas de veículos correspondem a 60% dos anúncios com leads.
	![image](https://user-images.githubusercontent.com/85720162/182040127-c903c926-e9e2-4859-b145-42c6806b6684.png)
	![image](https://user-images.githubusercontent.com/85720162/182040144-4ad3e639-b437-4852-80e2-85cab3823586.png)
	- Hipótese 03 [Verdadeira] - Veículos com ano de fabricação inferior a 2010 geram até 20% dos leads.
	![image](https://user-images.githubusercontent.com/85720162/182040214-53459251-0c8b-4d0d-bea3-cf902e750ecd.png)
	- Hipótese 04 [Verdadeira] - Veiculos Flex correspondem a no mínimo 60% dos leads.
	![image](https://user-images.githubusercontent.com/85720162/182040245-c5c60212-d765-426d-a352-f100fe2bae62.png)
	- Hipótese 05 [Verdadeira] - São Paulo concentra 80% dos anúncios que possuem mais de 30 leads.
	![image](https://user-images.githubusercontent.com/85720162/182040299-931bae46-96d0-4677-b636-2e9a67784834.png)
	![image](https://user-images.githubusercontent.com/85720162/182040330-b3895faa-7415-437b-a34e-e56265c07b96.png)
	- Hipótese 09 [Verdadeira] - Anúncios em que o anunciante é PJ correspondem a 60% dos que recebem leads.
	![image](https://user-images.githubusercontent.com/85720162/182040414-d5520a5b-705e-4462-933e-03f3853b5fd5.png)
	
05. **Data Preprocessing:** Nessa etapa, os dados foram preparados para posterior aplicação de algoritmos de Machine Learning. Foi aplicado o Feature Scaling nas variáveis quantitativas; encoding nas variáveis qualitativas; e foi realizado a balanceamento dos dados com a técnica SMOTE Tomek Link. 
 
06. **Feature Selection:** Nessa etapa, foi realizado um filtro nas features do dataset para aplicarmos os algoritmos de Machine Learning somente em dados que tenham a maior probabilidade de gerar bons resulatos. Para a seleção, foram extraídas as features mais importantes através de um modelo e, adicionado features identificadas como interessantes na análise exploratória dos dados. Acesse o notebook de desenvolvimento para verificar as features escolhidas.

### Modeling
07. **Machine Learning:** Nessa etapa, foram definidas as métricas de avaliação, treinado os modelos, realizado o Cross-Validation e, realizado a tunagem dos hiperparâmetros.
	
	- **Métricas:** A Acuracia será a principal métrica utilizada para avaliar os modelos, dado que as classes foram balanceadas na etapa de pré-processamento. As métricas a seguir também serao analisadas para complementar a análise dos resultados: Precision, Recall, F1-Score e ROC_AUC Score.
	
	- **Modelos:** Foram aplicados os algoritmos Naive Bayes, Random Forest, LightGBM.
	
	- **Cross-Validation:** Foi aplicado o CV com 15 folds e 5 repetições para validar o resultado do modelo com os dados de treino. **O modelo cirado a partir do LightGBM foi o escolhido nesse primeiro ciclo, pois, apresentou o melhor resultado, realizou o treinamento no menor tempo e o tamanho do modelo treinado ficou em 0.66 MB.**
	![image](https://user-images.githubusercontent.com/85720162/182041354-18675490-5eee-4625-b78d-096442f88e1d.png)

	
	- **Hyperparameter Fine Tuning:** Foi aplcado a técnica de otimização bayesiana para achar os hiperparametros do modelo. Porém, o modelo tunado não apresentou melhoras significantes nos resultados já obtidos anteriormente.
	![image](https://user-images.githubusercontent.com/85720162/182041423-ba537e83-4c75-45a4-9158-0daab5017909.png)
	![image](https://user-images.githubusercontent.com/85720162/182041439-c5da60e5-b414-4eee-bc5d-65496fd68e4f.png)

### Evaluation

08. **Performance:** Nessa etapa, todo o pipeline de transformações foi aplicado nos dados de teste para validar o desempenho do modelo. Porém, os resultados nesse primeiro ciclo foram muito ruins se comparado ao desempenho sobre os dados de treino, nos próximos ciclos será necessário se aprofundar um pouco mais nas análises para lapidar melhor os dados, gerar features mais relevantes e otimizar as transformações realizadas.
![image](https://user-images.githubusercontent.com/85720162/182041685-4b0594b2-6db8-4199-8ef7-6855ed8dcac9.png)

### Próximos Passos & Considerações
**Próximos Passos:**

- Analisar em detalhes os NAs que vieram originalmente no dataset para verificar se é possível inserir valores com base em alguma técnica estatística ou com um viés de negócio, buscando entender associações, relações e conexões entre as features.
- Analisar em detalhes as informações divergentes encontradas na análise das estatísticas descritivas.
- Criar mais hipóteses de negócios com o objetivo de aumentar os conhecimentos dos dados disponíveis.
- Criar mais features e que modelem melhor o fenômeno.
- Prosseguir para a análise das outras features do dataset que não foram verificadas no primeiro ciclo por falta de tempo.
- Realizar a análise multivariada para verificar as relações entre todas as features do dataset.
- Realizar o split entre treino e validação antes das transformações de escala e encoding para evitar data leakage (não realizado por falta de tempo).
- Testar outras técnicas de Reescaling para cada feature (não testado por falta de tempo).
- Testar outras técnicas de Encoding para cada feature (não realizado por falta de tempo).
- Se as demais alterações não resultarem em boas métricas, alterar a técnica dde balanceamento de dados para erificar se exista algum melhora.
- Aplicar outras técnicas de Feature Selection como Boruta e Recursive Feature Elimination, ou gerar a feature importance com outros algoritmos.
- Simplificar o modelo utilzado para não gerar Overfitting.
- Adicionar mais algoritmos para realizar o treinamento, como Logistc Regression, KNN, SVM, etc.
- Refazer todos os passos de forma cíclica e se não gerar resultados, verificar com o time de engenheiros de dados a possibilidade de obter novas features para modelar melhor o fenômeno.
- Após a generalização dos resultados, criar uma API para testar o envio de requisições e colocar a solução em produção.
	
**Considerações Finais:** Gostei muito de desenvolver o projeto de classificação, pois já conhecia a empresa e sua área de atuação, o que fez me envolver no projeto realizado. Conheci e apliquei uma técnica nova de balanceamento (SMOTE Tomek Link), passei por todas as etapas de desenvolvimento e acredito que tenha conseguido gerar insights acionáveis para os anunciantes. Apesar do resultado final do modelo nos dados de teste não terem sido satisfatórios nesse primeiro ciclo, pelo pouco tempo dedicado para realizar o projeto (2 dias), estou muito satisfeito com o resultado final apresentado e acredito que tenha chegado no meu objetivo para esse primeiro momento. *O Projeto de regressão não foi possível ser feito devido ao tempo disponível. 
	
