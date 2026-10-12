# Manga Task

Zero-shot baseline for manga dialogue extraction using Qwen2.5-VL-7B-Instruct.

## CREDITS
- Model: Qwen2.5-VL (Alibaba Qwen team, Hugging Face)
- Libraries: `transformers`, `qwen-vl-utils`, `bitsandbytes`
  

## My contributions
How did I design this model?
### Experiment 1-
  - I used the [Qwen-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen-VL-7B-Instruct) model as the base model.
#### The First Problem - 
  I dont have a very good gpu and hardware to train the model. So I used free GPUs from Kaggle to use this model. However, a 7-Billion parameter model normally requires around 16 GB to 28 GB of VRAM just to load, which exceeds the memory of a free Kaggle GPU (T4 with ~15 GB VRAM).
##### Soultion -
      I decided to shrink the model's footprint from 16 GB to 5 GB by using [bitsandbytes] NF4- 4 bit quantization. Certainly, this alone can degrade the performance of the model, but I think it is still a good trade-off for the model's size.

#### What I got-
  When I ran this model on kaggle, the results were not very satisfactory. 

  1. The model was still transcribing sound effects and other non-dialogue content.
  Eg. "SFX: HOOFBEATS" (in Silver Spoon)

  2. Even if, only 3-4 characters were speaking, the model was giving same speaker a new id.
  Eg. From Chainsaw Man- {
  "sequence_id": "seq_2fae297a18af2a57",
  "pages": [
    [
      { "speaker": "char1", "text": "GOT YOUR TOMATO DEVIL RIGHT HERE. IT REVIVES FROM ITS SEEDS, SO YOU SHOULD BURN IT." },
      { "speaker": "char2", "text": "GOOD WORK, DENJI. THIS GUY'S CORPSE'LL FETCH A PRETTY PENNY ON THE BLACK MARKET." }
    ],
    [
      { "speaker": "char4", "text": "AFTER PAYING THE WATER BILL AND THE OTHER LOANS, THERE'S A MEASLY 1800 YEN LEFT." },
      { "speaker": "char4", "text": "POCHITA... DINNER TODAY'S GONNA BE ONE SLICE OF BREAD." }
    ],
    [
      { "speaker": "char5", "text": "WHY'RE YOU USING A BRAT LIKE HIM AS A DEVIL HUNTER?" },
      { "speaker": "char6", "text": "TO GET BACK THE MONEY WE'RE OWED. THOUGH IT WAS HIS PIECE OF SHIT POPS WHO DIED WITHOUT PAYING UP." }
    ]
  ]
}
  
  3. Somtimes, even different characters got the same speaker id.
  Eg. [
  { "speaker": "char1", "text": "NOW... HOW WILL YOU REACT...?" },
  { "speaker": "char1", "text": "SHINOMIYA THOUGHT FOR AN INSTANT..." },
  { "speaker": "char1", "text": "HE'S ATTACKED ME..." },
  { "speaker": "char1", "text": "BY REVEALING THAT HE WANTS TO SEE THE MOVIE..." }
]
  
  4. Pretraining of model became an issue here! Because Qwen2.5-VL was pretrained on millions of standard Western web pages and documents, its natural instinct was to read the top-left panel first. On complex, multi-tiered manga pages, it frequently read dialogues backwards, causing a low text_order_score...
  
  This led me to my next experiment. Fine Tuning !!!

### Experiment 2-
    This time, I decided to use Qlora to fine-tune the model.
    We know that traidionally we need to update every parameter in the model but here we use tiny adapters to update only the parameters that are important for the task. They make up only 0.6% of the model's parameters. This allowed us to cut down meory requirements.

    I used standerd ml ideas to split data into 64:16 train:validation.
     
    The model was simply asked to convert the training page into text and compared the weights to the ground truth. Pytorch simply calculated the loss between the weights and the ground truth and tried to improve the model. 

    However, when I actually tried to train the model, at our initial resolution setting (max_pixels: 802816), the vision encoder broke those images down into 5,000+ visual tokens. Due to this, it needed an extra 2.3 GB of VRAM.

#### Solution -
    I simply decided to reduce the max_pixels from 802816 to 313600. This allowed the model to use less VRAM.

### Experiment 4-
    So, when I tried to train the model, I actually managed to train first epoch. But, then the model tried to validate the 16 images in the validation set. Since the batch size was restricted to 8, it loaded 24 images(8 manga sets) at a time. This led the program to crash.

#### Solution -
    I decided that I will first train all of my data and than later validate it.
  
### Experiment 5-
    This time, I actually manged to train the model. 

    However, I had not done in training validation. So, I first set the code to process 1 image at a time and do it for all 80 sequences. Then, I created a file for predictions.jsonl for all of them. 

    Than, I identified those 16 sequenses and took thier answers only and compared them with the ground truth.

    The ocr and reading accuracy was 60.4%. 

    The speaker accuracy was 55.5%

    The text order accuracy was 58.4%

    The joint F1 score(where the model managed to do all things right) was 38.6%.

#### Reasons for lower scores - 
    1. I had already reduced the max_pixels to 313600 from 802816. This made the image panels unclear for the model.

    2. The training data is small and the free GPUs I used were not powerful enough.