import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import re
import pdfplumber


# Large built-in skill set
SKILL_DB = [
    # Programming & Scripting Languages
    "python","java","c++","c#","javascript","typescript","ruby","php","swift","go",
    "rust","kotlin","scala","r","matlab","perl","lua","dart","objective-c","sql",
    "pl/sql","t-sql","bash","powershell","shell scripting","sas","stata","octave",
    "haskell","clojure","elixir","f#","julia","cobol","fortran","assembly","vhdl",
    "verilog","delphi","pascal","lisp","prolog","smalltalk","erlang","groovy",

    # Web & Frontend Frameworks
    "react","react.js","React.js","angular","angular.js","vue.js","node","express","next.js","nuxt.js","django","flask"," React.js",
    "spring boot","laravel","ruby on rails","bootstrap","tailwind css","material-ui",
    "jquery","html","css","html5","css3","sass","less","webpack","babel","alpine.js","ionic",
    "cordova","electron","svelte","react native",

    # Databases & Data Management
    "mysql","postgresql","mongodb","oracle","sqlite","redis","cassandra","elasticsearch",
    "dynamodb","firebase","hbase","sql server","bigquery","data warehouse","microsoft access",
    
    # Cloud Platforms & DevOps
    "aws","azure","google cloud","docker","kubernetes","jenkins","git","github","gitlab",
    "terraform","ansible","puppet","circleci","ci/cd","vagrant","helm","cloudformation",
    "openshift","mesos","rabbitmq","kafka","nifi","airflow","datadog","prometheus","grafana",
    "cloudwatch","splunk","new relic",

    # Data Science & Machine Learning
    "machine learning","deep learning","tensorflow","keras","pytorch","scikit-learn",
    "pandas","numpy","matplotlib","seaborn","plotly","power bi","tableau","qlikview",
    "excel","excel vba","data visualization","statistics","nlp","computer vision",
    "reinforcement learning","time series","predictive modeling","data analytics",
    "data mining","big data","hadoop","spark","pyspark","etl","power query",
    "huggingface","mlflow","opencv","onnx","fastai","xgboost","lightgbm","catboost",
    "streamlit","dash","mlops","data engineering","data governance","power pivot",

    # Analytical & Soft Skills
    "problem solving","critical thinking","analytical skills","logical reasoning",
    "communication","presentation","teamwork","project management","time management",
    "leadership","agile","scrum","kanban","risk management","stakeholder management",
    "business analysis","requirements gathering","product management","negotiation",
    "conflict resolution","mentoring","networking","emotional intelligence","adaptability",
    "creativity",

    # Testing & Quality
    "selenium","cypress","junit","pytest","robot framework","manual testing",
    "automation testing","performance testing","load testing","unit testing",
    "integration testing","qa","qa automation","test plan","test cases",

    # Other Tools & Technologies
    "sap","erp","crm","salesforce","unix","linux","windows server","gitbash","vim",
    "vscode","intellij","eclipse","android studio","xcode","visual studio","notepad++",
    "jira","confluence","asana","trello","monday.com","tableau prep","power automate","qlik sense",

    # Security / Networking
    "penetration testing","wireshark","nmap","cisco","palo alto","fortinet","aws security",
    "cryptography","firewall configuration","tcp/ip","vpn",

    # Design / Creative / UX Tools
    "photoshop","illustrator","figma","adobe xd","blender","maya","sketch","after effects",
    "premiere pro","final cut pro","cinema 4d","corel draw","audition","lightroom",

    # Miscellaneous / Domain-Specific Skills
    "sap s/4hana","oracle erp cloud","fintech","blockchain","solidity","smart contracts",
    "iot","ar/vr","robotics","embedded systems","3d printing","digital marketing","seo",
    "sem","google analytics","ethereum","hyperledger","rpa","uipath","uipath studio",
    "blue prism","spark ar studio","unity","unityhub","unreal engine","autocad","solidworks",
    "3ds max","sketchup","vuforia",

    # Environmental / Sustainability
    "environmental science","sustainability","renewable energy","solar energy",
    "wind energy","climate change","carbon footprint analysis",

    # Social Media & Marketing
    "social media management","content marketing","facebook ads","instagram ads",
    "linkedin marketing","email marketing","hubspot","canva","hootsuite",

    # Sports & Extracurricular
    "football","cricket","basketball","tennis","yoga","swimming","marathon","gym",
    "chess","public speaking","debating","volunteering","team sports","coaching",
    "badminton","table tennis","cycling","running","hiking","meditation","mindfulness",
    "photography","painting","music","singing",

    # Interior / Architecture
    "interior design","landscape design","space planning","revit","feng shui",
    "lighting design","3d modeling",

    # Automation / Industrial
    "industrial automation","plc programming","scada","labview","embedded c","mechatronics",

    # Finance / Accounting / Legal
    "accounting","bookkeeping","quickbooks","tax preparation","financial modeling","auditing",
    "corporate law","contract management","risk assessment",

    # Education / Training
    "curriculum design","e-learning","lesson planning","teaching","training",
    "instructional design","coaching",

    # Healthcare / Life Sciences
    "cpr","patient care","clinical research","medical coding","lab techniques",
    "pharmaceutical research","biostatistics"

    # Data Science & Machine Learning
    "data wrangling","feature engineering","model evaluation","hyperparameter tuning",
    "ensemble methods","neural networks","natural language processing","computer vision",
    "reinforcement learning","time series forecasting","anomaly detection","dimensionality reduction",
    "data pipelines","big data analytics","cloud-based ML platforms","model interpretability",
    "AI ethics","bias mitigation","explainable AI","automated ML","data augmentation",
    "transfer learning","deep reinforcement learning","generative AI","graph neural networks",
    "federated learning","meta learning","autoencoder","variational autoencoder","GANs",
    "self-supervised learning","semi-supervised learning","unsupervised learning","supervised learning",
    "feature selection","feature extraction","predictive maintenance","recommendation systems",
    "object detection","image segmentation","speech recognition","text summarization","chatbot development",
    "topic modeling","sentiment analysis","knowledge graph","time series anomaly detection","outlier detection",
    "causal inference","statistical modeling","probabilistic modeling","bayesian inference","monte carlo simulation",
    "optimization algorithms","constraint programming","linear regression","logistic regression","decision trees",
    "random forest","gradient boosting","xgboost","lightgbm","catboost","support vector machines",
    "k-means clustering","hierarchical clustering","DBSCAN clustering","principal component analysis",
    "independent component analysis","t-SNE","UMAP","matrix factorization","latent semantic analysis",
    "natural language understanding","language modeling","BERT","GPT","transformers","tokenization","word embeddings",
    "doc2vec","word2vec","glove","fastText","attention mechanisms","sequence modeling","recurrent neural networks",
    "long short-term memory","gated recurrent units","convolutional neural networks","residual networks","alexnet",
    "vggnet","mobilenet","efficientnet","squeezenet","inceptionnet","object tracking","pose estimation",
    "3D reconstruction","point cloud processing","LiDAR processing","video classification","motion detection",
    "human activity recognition","optical character recognition","image captioning","text-to-speech","speech-to-text",
    "audio classification","audio signal processing","music generation","emotion recognition","behavior analysis",
    "robotics perception","robotics control","path planning","reinforcement learning for robotics","multi-agent systems",
    "simulation environments","Gazebo simulation","ROS","robot kinematics","robot dynamics","robot manipulation",
    "robot localization","SLAM","autonomous navigation","autonomous vehicles","computer vision pipelines",
    
    # Web & Frontend Development
    "progressive web apps","single-page applications","responsive design","API integration",
    "server-side rendering","client-side rendering","state management","redux","mobx",
    "vuex","react hooks","react context","component lifecycle","frontend testing","jest",
    "mocha","cypress testing","enzyme testing","storybook","tailwind config","postcss",
    "CSS Grid","CSS Flexbox","media queries","cross-browser compatibility","web accessibility",
    "WCAG compliance","SEO best practices","performance optimization","lazy loading","code splitting",
    "service workers","offline caching","PWA manifest","web push notifications","graphQL","REST API",
    "API versioning","token-based authentication","OAuth","JWT","cookie management","session management",
    "frontend security","XSS prevention","CSRF prevention","CSP implementation","web sockets","real-time communication",
    "socket.io","webRTC","video streaming","audio streaming","drag-and-drop UI","canvas animations","SVG animations",
    "Three.js","D3.js","chart.js","highcharts","plotly.js","interactive dashboards","data visualization web apps",
    
    # Backend & Databases
    "REST API design","GraphQL API design","microservices architecture","monolithic architecture",
    "serverless architecture","AWS Lambda","Azure Functions","Google Cloud Functions","API documentation",
    "OpenAPI","Swagger","Postman","database indexing","query optimization","stored procedures","triggers",
    "views","materialized views","database replication","database sharding","database partitioning",
    "ACID transactions","CAP theorem","NoSQL design","document database","key-value store","columnar database",
    "graph database","Neo4j","OrientDB","Cassandra data modeling","MongoDB aggregation","Elasticsearch queries",
    "Redis caching","memcached","database backup","database recovery","SQL tuning","data modeling","ER diagrams",
    
    # Cloud & DevOps
    "infrastructure as code","Terraform modules","Ansible playbooks","Puppet manifests","Helm charts",
    "Kubernetes deployment","Docker Compose","containerization","container orchestration","CI/CD pipelines",
    "Jenkins pipelines","GitHub Actions","GitLab CI","CircleCI configuration","Travis CI","Azure DevOps",
    "AWS CodePipeline","monitoring and alerting","Prometheus metrics","Grafana dashboards","Datadog monitoring",
    "Splunk logging","ELK stack","CloudWatch logs","incident response","disaster recovery planning","load balancing",
    "auto scaling","service mesh","Istio","Linkerd","Nginx configuration","HAProxy","cloud security","IAM management",
    "encryption at rest","encryption in transit","network security","VPC peering","subnet design","firewall rules",
    
    # Security & Networking
    "penetration testing","vulnerability assessment","metasploit","nmap scanning","Wireshark analysis",
    "Burp Suite","OWASP Top 10","XSS testing","SQL injection testing","CSRF testing","network segmentation",
    "VPN setup","firewall configuration","IDS/IPS","SIEM systems","cryptography","AES encryption","RSA encryption",
    "elliptic curve cryptography","hashing algorithms","digital signatures","public key infrastructure","TLS/SSL",
    
    # Design & Creative
    "UI design","UX design","wireframing","prototyping","user research","usability testing",
    "Figma components","Sketch symbols","Adobe XD prototyping","motion graphics","animation","3D modeling",
    "Blender rendering","Maya animation","Unity game development","Unreal Engine development","game mechanics",
    "AR development","VR development","interactive storytelling","digital illustration","graphic design",
    "branding","typography","color theory","layout design","print design","packaging design",
    
    # Business, Management & Soft Skills
    "project management","agile methodology","scrum master","product owner","kanban management",
    "risk assessment","stakeholder engagement","budget management","resource allocation","conflict resolution",
    "leadership","team building","mentoring","coaching","negotiation","strategic planning","business analysis",
    "process improvement","lean methodology","six sigma","customer relationship management","market research",
    "competitive analysis","business strategy","digital marketing strategy","content strategy","social media strategy",
    "SEO strategy","PPC campaign management","email campaign optimization","brand management","sales strategy",
    "financial modeling","budget forecasting","investment analysis","cost reduction","risk management","compliance",
    
    # Education & Training
    "learning management systems","LMS administration","e-learning content creation","instructional design",
    "curriculum planning","assessment design","teacher training","student engagement strategies","blended learning",
    "gamified learning","adaptive learning platforms","virtual classrooms","webinar facilitation","online tutoring",
    
    # Healthcare & Life Sciences
    "clinical trials","medical research","pharmaceutical development","lab safety protocols","GxP compliance",
    "patient data management","clinical data analysis","biostatistics in healthcare","epidemiology","genomics",
    
    # Emerging Technologies
    "blockchain development","smart contract development","Solidity programming","Ethereum DApps",
    "Hyperledger Fabric","RPA automation","UiPath workflows","Blue Prism processes","IoT device integration",
    "edge computing","quantum computing basics","3D printing design","digital twin modeling","AR/VR simulations",
    "metaverse development","chatbot AI","voice assistants","computer vision for AR","MLops pipelines",
    
    # Productivity & Office Tools
    "Microsoft Office advanced","Excel formulas","Excel macros","Excel dashboards","PowerPoint design","Google Workspace",
    "Google Sheets automation","Notion productivity","Trello project tracking","Asana workflows","Slack communication",
    
    # Miscellaneous
    "photography editing","video editing","content creation","public speaking","debating","volunteering",
    "sports coaching","fitness training","yoga instruction","mindfulness coaching","language translation","copywriting",
    "technical writing","creative writing","storytelling","podcast production","event management","fundraising","community outreach"
]


STOPWORDS_for_MUM = ["is","the","and","with","a","an","of","for","on","in","to","any","that","from",
    "as","by","at","be","this","these","those","are","was","were","will","would",
    "can","could","should","may","might","have","has","had","do","does","did","doing",
    "about","above","after","again","against","all","am","also","among","amount","another",
    "anybody","anyone","anything","anywhere","back","because","been","before","being",
    "below","between","both","but","cannot","come","comes","could","did","do","does",
    "doing","down","during","each","either","else","elsewhere","ever","every","everybody",
    "everyone","everything","everywhere","few","first","for","former","formerly","from",
    "further","get","gets","getting","give","given","gives","go","goes","going","gone",
    "got","had","has","have","having","he","hence","her","here","hereafter","hereby",
    "herein","hereupon","hers","herself","him","himself","his","how","however","i",
    "ie","if","in","inc","indeed","into","is","it","its","itself","just","keep","keeps",
    "kept","latter","latterly","less","like","likewise","may","me","meanwhile","might",
    "more","moreover","most","mostly","much","must","my","myself","namely","neither",
    "never","nevertheless","next","no","nobody","none","noone","nor","not","nothing",
    "now","nowhere","of","off","often","on","once","one","only","onto","or","other",
    "others","otherwise","our","ours","ourselves","out","over","own","part","per","perhaps",
    "please","put","rather","re","same","see","seem","seemed","seeming","seems","serious",
    "several","she","should","show","side","since","so","some","somebody","someone","something",
    "somewhere","still","such","take","than","that","the","their","theirs","them","themselves",
    "then","thence","there","thereafter","thereby","therefore","therein","thereupon","these",
    "they","thick","thin","this","those","though","through","throughout","thru","thus","to",
    "together","too","toward","towards","under","unless","until","up","upon","us","very","via",
    "was","we","well","were","what","whatever","when","whence","whenever","where","whereafter",
    "whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither",
    "who","whoever","whole","whom","whose","why","will","with","within","without","would",
    "yet","you","your","yours","yourself","yourselves"
    "ability","able","about","above","abroad","absence","absolute","absolutely","abstract","accept",
    "accepted","access","according","account","achieve","achieved","across","act","action","active",
    "actively","activity","actual","actually","ad","add","added","adding","addition","additional",
    "address","adequate","adjacent","adjust","admin","administration","adopt","advance","advanced","advantage",
    "advice","advise","affect","affiliated","afterwards","age","agency","agenda","agent","agreement",
    "ahead","aid","aim","air","albeit","allow","allowed","allowing","along","already","alternative",
    "although","always","amend","analysis","analyze","announce","annual","answer","anticipate","anyway",
    "apart","apparent","apparently","apply","applicable","applicant","application","approach","appropriate",
    "approval","approve","area","argue","argument","arise","around","arrange","arrangement","art","article",
    "ask","aspect","assess","assessment","asset","assign","assigned","assignment","assist","assistance",
    "associate","associated","assume","assumption","assure","attach","attached","attempt","attend","attention",
    "attitude","attract","attribute","audience","author","authority","available","average","avoid","aware",
    "awareness","away","background","balance","bank","base","based","basis","bear","beat","become","becomes",
    "begin","beginning","behavior","behind","believe","benefit","beside","best","better","beyond","bid","big",
    "bill","bit","board","body","book","border","bottom","break","brief","bring","brought","budget","build",
    "building","bulk","business","busy","buy","call","called","calling","capacity","capital","capture","care",
    "career","careful","carefully","carry","case","cash","catch","cause","cease","center","central","certain",
    "certainly","chair","challenge","chance","change","character","charge","check","chief","child","choice",
    "choose","chosen","church","circumstance","cite","citizen","city","civil","claim","clarify","class","clean",
    "clear","clearly","client","close","closely","coach","code","coffee","cold","collaborate","collaboration",
    "collect","collection","college","color","commit","commitment","committee","common","communicate","communication",
    "community","company","compare","comparison","compete","competence","competent","complete","completely",
    "complex","component","comprehensive","compute","computer","concern","concerned","conclude","conclusion",
    "condition","conduct","conference","confidence","confirm","connected","consequence","consider","considerable",
    "consideration","consistent","constant","constantly","construct","construction","consult","consultant","consume",
    "contact","contain","content","continue","contract","contrast","control","convenient","conversation","convert",
    "convince","cook","cool","corporate","correct","correctly","cost","council","count","country","couple","course",
    "court","cover","create","created","creating","creation","creative","credit","criteria","critical","critically",
    "cross","current","currently","customer","cut","cycle","daily","damage","data","date","deal","decide","decision",
    "declare","decrease","dedicated","deep","defend","define","defined","definition","degree","delay","deliver",
    "delivery","demand","demonstrate","demonstrated","denote","department","depend","depending","deploy","deployment",
    "describe","described","description","design","designate","desire","desk","detail","detailed","detect","determine",
    "develop","developed","developing","development","device","difference","different","difficult","direct","direction",
    "directly","director","discuss","discussion","display","distribute","distribution","diverse","diversity","divide",
    "division","document","domain","domestic","door","double","down","draft","draw","drawing","drive","drop","during",
    "early","earn","ease","easily","east","easy","economic","economy","edge","education","effect","effective",
    "effectively","effort","elect","electronic","element","elsewhere","emerge","emergency","employee","employer",
    "employment","enable","encourage","end","energy","enforce","engage","engineer","engineering","enhance","enjoy",
    "enough","ensure","enter","enterprise","entire","environment","equipment","equivalent","error","especially",
    "essential","establish","established","etc","evaluate","evaluation","even","event","eventually","evidence",
    "exact","exactly","exam","examine","example","excellent","except","exchange","execute","executed","exercise",
    "exist","existence","expand","expect","expectation","expense","experience","experienced","experiment","expert",
    "explain","explanation","explore","express","extend","extension","external","extra","extremely","eye","face",
    "facility","fact","factor","fail","failure","fair","fall","familiar","family","famous","far","farm","fast","fat",
    "favor","favorite","feature","federal","fee","feel","feeling","field","fight","figure","file","fill","film","final",
    "finally","finance","financial","find","finding","fine","finish","fire","firm","first","fish","fit","five","fix",
    "floor","flow","focus","follow","following","food","foot","force","foreign","forest","forget","form","formal",
    "forth","forty","forward","found","foundation","four","fourth","frame","freedom","frequent","frequently","fresh",
    "friend","friendly","front","full","fully","fun","function","fund","future","gain","game","garden","gas","general",
    "generally","generate","generation","global","goal","govern","government","grade","grant","great","green","ground",
    "group","grow","growing","growth","guarantee","guess","guide","guideline","hand","handle","hang","happen","happy",
    "hard","hardly","head","health","hear","heart","help","heritage","high","highlight","highly","hire","history","hold",
    "home","hope","hospital","hot","hour","house","human","hundred","identify","identity","image","immediate","impact",
    "implement","implementation","importance","important","improve","include","included","including","income","increase",
    "increased","increasing","indicate","individual","industry","information","initial","initiative","input","insight",
    "inside","instead","institution","instruction","instrument","insurance","intelligence","intended","interest","interested",
    "interesting","internal","international","internet","interpret","interview","introduce","introduction","invest","investigate",
    "investment","involved","issue","item","job","join","joint","judge","judgment","jump","keep","key","kill","kind","king",
    "know","knowledge","known","label","lack","large","largely","last","late","later","latest","law","lay","lead","leader",
    "leading","learn","learning","least","better","real","secure","twice"]



# Extract text from PDF
def extract_text_from_pdf_for_MUM(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    
    # Lowercase + remove punctuation (commas, periods, etc.)
    text = text.lower()
    text = re.sub(r'[^\w\s\.\+#-]', ' ', text)  # keep dots (for react.js), +, #, - but remove other punctuations

    # Split words & filter stopwords
    words = [w.strip() for w in text.split() if w and w not in STOPWORDS_for_MUM]
    return words

# Match skills using SKILL_DB (only from JD)
def match_skills_for_MUM(resume_words, job_description):
    jd_words = [w.lower() for w in job_description.replace(",", " ").split()]

    # Skills relevant only to JD
    jd_skills = {skill for skill in SKILL_DB if skill in jd_words}

    matched = {skill for skill in jd_skills if skill in resume_words}
    unmatched = jd_skills - matched  # JD skills not found in resume

    matched_list = list(matched) if matched else ["None of the skills match"]
    unmatched_list = list(unmatched) if unmatched else ["All skills are match"]

    return matched_list, unmatched_list

# Process multiple resumes (from Streamlit uploads)
def process_resumes_for_MUM(uploaded_files, job_description):
    result = {}
    for file in uploaded_files:
        resume_words = extract_text_from_pdf_for_MUM(file)  # pass file object directly
        matched, unmatched = match_skills_for_MUM(resume_words, job_description)
        result[file.name] = {"match_skilled": matched, "unmatch_skilled": unmatched}
    return result


# ---------- PDF Extraction ----------
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ---------- Text Preprocessing ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[\W_]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Extract Skills from Job Description ----------
def extract_skills_from_text(text):
    stopwords = set([
        "and", "or", "with", "in", "on", "for", "to", "the", "of", "a", "an", "is",
        "are", "as", "by", "at", "from", "that", "this", "be", "will", "have"
    ])
    words = preprocess_text(text).split()
    unique_words = sorted(set(words) - stopwords)
    return unique_words

# ---------- Skills Matching ----------
def match_skills(uploaded_files, resumes, skill_list):
    results = {}
    for file, resume_text in zip(uploaded_files, resumes):
        resume_text_processed = preprocess_text(resume_text)
        resume_words = set(resume_text_processed.split())

        matched = []
        unmatched = []

        for skill in skill_list:
            if skill in resume_words:
                matched.append(skill)
            else:
                unmatched.append(skill)

        candidate_name = file.name.rsplit(".", 1)[0]  # remove .pdf extension
        results[candidate_name] = {
            "matched_skills": matched,
            "unmatched_skills": unmatched
        }

    return results

# ---------- Resume Ranking ----------
def rank_resumes(job_description, resumes):
    job_description_processed = preprocess_text(job_description)
    resumes_processed = [preprocess_text(resume) for resume in resumes]
    documents = [job_description_processed] + resumes_processed

    vectorizer = TfidfVectorizer(stop_words='english').fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    scores = (cosine_similarities * 100).round(2)

    return scores

# ---------- Streamlit UI ----------
def render_resume_filter():
    st.title("Resume Filter")
    st.write("This tool helps you screen resumes against a job description using AI.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Job Description")
        job_description = st.text_area("Enter the job description", placeholder="Enter Job Description ...", height=150)

        st.subheader("📂 Upload Resumes")
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        # st.subheader("💡 Skills List (optional, comma separated)")
        skills_input = "" #st.text_area("Enter skills (optional)", "")

    with col2:
        if uploaded_files and job_description:
            st.subheader("📊 Ranking Resumes")

            # Extract skills from job description
            skill_list_from_jd = extract_skills_from_text(job_description)

            # Merge manual skills if given
            if skills_input.strip():
                manual_skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]
                skill_list = sorted(set(skill_list_from_jd + manual_skills))                
            else:
                skill_list = skill_list_from_jd

            # Ranking scores
            resumes = [extract_text_from_pdf(file) for file in uploaded_files]
            scores = rank_resumes(job_description, resumes)

            # Skills Matching
            skills_result = process_resumes_for_MUM(uploaded_files, job_description)

            # ✅ Build dataframe for skills + scores
            results_data = []
            for idx, file in enumerate(uploaded_files):
                file_name = file.name
                matched = ", ".join(skills_result[file_name]["match_skilled"])
                unmatched = ", ".join(skills_result[file_name]["unmatch_skilled"])
                score = scores[idx]
                results_data.append([file_name, matched, unmatched, score])

            results_df = pd.DataFrame(results_data, columns=["Resume", "Matched Skills", "Unmatched Skills", "Score"])
            results_df = results_df.sort_values(by="Score", ascending=False)


             # Chart
            fig = px.bar(results_df, x="Resume", y="Score", text="Score",
                         labels={"Score": "Match Score"}, color="Score",
                         color_continuous_scale=px.colors.sequential.Plasma)
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title="Resume", yaxis_title="Match Score", yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)    

            # Best Match
            best_resume = results_df.iloc[0]
            st.success(f"🏆 **Best Match:** {best_resume['Resume']} with a score of {best_resume['Score']:.2f}")

            st.markdown("### 📑 Resume Skills Match")
            st.dataframe(results_df, height=400)

                  

            # JSON (optional debugging view)
            # st.markdown("### 🧠 Raw Skills Match JSON")
            # st.json(skills_result)

            

        elif not job_description:
            st.info("ℹ️ Please enter a job description.")
        elif not uploaded_files:
            st.info("📂 Please upload one or more resume PDFs.")

# Run the Streamlit App
if __name__ == "__main__":
    render_resume_filter()
