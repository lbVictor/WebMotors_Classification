## WEB MOTORS CLASSIFICATION
![image](https://user-images.githubusercontent.com/85720162/182034807-7f59fb8c-3c15-4c6b-a008-c6675526dba6.png)

------------------------------------------------------------------
## Sobre o Projeto
Esse projeto foi realizado como teste de conhecimento em um case técnico, os dados utlizados são fictícios e foram disponibilizados pela empresa. O objetivo desse projeto é extrair insights que possam ajudar anúnciantes a anúnciarem melhore seus veículos na plataforma e construir um modelo de Machine Learning para classificar se um anúncio vai/não gerar leads.

O projeto foi realizado de forma cíclica (metodologia CRISP-DM), sendo cada iteração uma melhoria do que foi desenvolvido anteriormente.

**Esse projeto representa o primeiro ciclo da solução.**

Acesse o notebook de desenvolvimento do projeto nesse link: 
https://github.com/lbVictor/Health_Insurance_Cross_Sell/blob/main/development/notebooks/health_insurance_cross_sell_cycle01.ipynb

## Sobre a Empresa 
Webmotors é uma plataforma de anúncios, onde compradores encontram vendedores de veículos. Não é possível comprar ou vender diretamente no site da Webmotors. Ou seja, o site é um intermediário. Assim, o interessado pode entrar em contato com quem disponibiliza o produto. Depois disso a negociação e a compra serão efetuadas fora do site. Para o comprador é uma ótima maneira de procurar seu carro ideal sem sair de casa, além de poder comparar preços. Já para o vendedor é uma oportunidade de ter uma grande alcance de público que está interessado em um automóvel igual ao dele.


## Estrutura do Projeto
### Entendimento do Negócio
00. **Problema de Negócio:** A Webmotors dispõe de diversos produtos digitais para solucionar problemas durante a compra, utilização e venda de um veículo, sendo um de seus principais e mais conhecidos o classificado. Para o classificado utilizamos o lead model para monetizar nosso produto. Um lead é uma demonstração de interesse de um comprador para um vendedor. 

    **Desafio Proposto:** A fim de melhorar o desempenho dos anúncios, determine ao menos uma alternativa para potencializar o recebimento de leads dos anúncios. Para tal, elaboramos um roteiro em duas partes para te guiar neste desafio:

      **Parte 1:** Faça uma exploração dos dados e mostre como a quantidade de leads varia de acordo com as outras variáveis.
      
      **Parte 2:** Proponha um modelo para determinar se um anúncio receberá lead. Que outras informações, as quais você não teve acesso, poderiam ajudar a chegar a conclusões melhores?

      **Desafio Bônus:** Proponha um modelo para determinar a quantidade de leads que um anúncio irá receber

	
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

03. **Data Cleaning & Data Description:** Nessa etapa, foi realizado a divisão do conjunto de dados entre Treino e Teste; analisado a descrição das features; verificado o tamanho do conjunto de dados; verificado dados duplicados; verificado valores faltantes (devido ao tempo disponibilizado para esse primeiro ciclo, os valores faltantes do dataset foram removidos); verificado e alterado o tipo dos dados; realizado uma análise estatística descritiva para avaliar os dados categóricos e númericos e definir ações necessárias para limpeza ou análises posteriores.

04. **Feature Engineering:** Nessa etapa, foi desenvolvido um Mapa Mental para relacionar as caracteríticas do fenômeno de geração de leads pela Webmotors, com o objetivo de: identificar features interessantes para serem analisadas; criar hipóteses de negócio que possam vir a gerar insights acionáveis; conduzir a EDA; facilitar o cruzamento de informações para realizar o Feature Engineering.
         
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
          03. Qauntidade média de leads por estado.
          04. Quantidade de itens opcionais de fábrica por veículo anunciado.
          05. Variável binária informando se o IPVA foi pago E o Licenciamento está em dia.
          06. Variável binária informando se o valor do anúncio é inferior/igual ao valor de mercado, ou se é superior.
          


05. **Data Filtering:** Nessa etapa, foi excluído 

At this stage, the objective is to analyze business limitations, data with wrong values or unnecessary columns to filter the dataset, but none of these changes were necessary.

06. **Exploratory Data Analysis (EDA):** At this stage, three types of analysis were performed in order to better understand the available data.
	* **Univariate analysis:** carried out in order to understand the individual behavior of each variable.
	* **Bivariate analysis:** carried out in order to understand the relationship of some features with the response variable through the validation of the raised hypotheses.
	* **Multivariate Analysis:** carried out in order to understand the relationship/correlation between all features + response variable.
	
07. **Data Preprocessing:** At this stage, the preparation of data for future application in machine learning algorithms was performed. The objective is to adjust the data without losing the information content in order to facilitate its understanding by machine learning algorithms.
	* **Rescaling:** For numerical variables, MinMax Scaler was applied in features without outliers ans Robust Scaler was applied in features with outliers.
	* **Encoding:** For categorical variables, Label Encoding, Ordinal Encoding and Frequency Encoding were applied according to the characteristics of each feature. 
		 
08. **Feature Selection:** In this step, feature importances from LightGBM algorithm and Random Forest were obtained to join with the knowledge obtained at EDA and choose which features would be used to perform the training of the models.


### Modeling
09. **Machine Learning:** In this step, the evaluation metric was defined; seven evaluation algorithms were trained (simple and sophisticated algorithms were applied to evaluate the results and verify the complexity of the phenomenon); cross-validation was performed to obtain the real results of the model; the hyperparameter fine tuning was performed to obtain the best parameters for the chosen model.
	
    - **Precision@k Metric**: The @k metrics are used when we want to apply a metric limiting the examples to a value (k). After generating the purchase propensity, the dataset was sorted from the customer with the highest propensity to the lowest and I used as a value k the amount of people interested in the dataset in which I was applying the model (validation or testing), so customers who are up to that value are the customers that my model predicted to be most interested in vehicle insurance. A precision @k of 100% means that all of my model's predictions were right, 50% means that up to the @k value my model was right 50% of the time, etc.
  
  
    - **Machine Learning Results with Cross-Validation and the size of the trained model:**
	
      ![image](https://user-images.githubusercontent.com/85720162/153069641-8fa6c186-ad43-4dfc-898d-a31385b31a08.png)
      
      Based on the results obtained by the models, **LightGBM** was chosen because it presented a very satisfactory result in relation to the others and the size of the trained model is the smallest among those with good results.

    - **Hyperparameter Fine Tuning:** Bayesian Optimization was used to obtain the best parameters for the chosen model.
	
	    **LightGBM Final Result:**
      
	    <img src="https://user-images.githubusercontent.com/85720162/153072721-58df50a6-ab99-4964-a4f7-1a036d8736de.png" width="120">

	
###	Evaluation

10. **Performance:** The performance of the model was evaluated from a Machine Learning perspective and from a business perspective to verify the final result of the model and its impact on the business.

  * **Machine Learning Performance:**

    - **Precision@k:**
  
      <img src="https://user-images.githubusercontent.com/85720162/153074144-df928a95-e1b9-46ae-b25e-91b135224695.png" width="200">


    - **Gain & Lift Curve:**
  
      <img src="https://user-images.githubusercontent.com/85720162/153074311-a6809e58-39a2-4701-a51c-b36bd8c1b883.png" width="700">
      
      As we can see in the precision@k and lift curve above, there was a small difference between the validation and test results, however, the result under the test data was 2.8x greater than the random result (without the propensity to purchase score, represented by the black line).
		
* **Business Performance:** 
 
  - **Business Questions Answered:**
    1. Percentage of customers interested in vehicle insurance in 20,000 calls
		![image](https://user-images.githubusercontent.com/85720162/153647610-564585f7-54c5-43fa-a477-a7f28ec56477.png)

    3. Percentage of customers interested in vehicle insurance in 40,000 calls
    		![image](https://user-images.githubusercontent.com/85720162/153647887-7de9c7f3-efc9-4521-8bd1-589f901eec6e.png)

    5. Number of calls needed to contact 80% of those interested in vehicle insurance
    		![image](https://user-images.githubusercontent.com/85720162/153648179-bacd2f31-a578-49f9-ad8b-cdc759773410.png)

	


###	Deploy
11. **Deploy Model to Production:** To upload the model into production, an application was created on Heroku Cloud containing an API with the entire pipeline of transformations necessary for the raw data to be eligible for the application of the stored model, which receives the data and returns the purchase propensity. To facilitate future forecasts by the sales team, a spreadsheet was created in Google Sheets integrated into the model, where it is possible to obtain the purchase propensity just by clicking on a button created through Google Scripts.


<p align="center">
Google Sheets Application 
</p>

<p align="center">
 	<img width="800" alt="drawing" src="https://user-images.githubusercontent.com/85720162/153077216-b2995c43-d0e0-424e-bda8-e0cb2441e17c.jpg">
</p>
