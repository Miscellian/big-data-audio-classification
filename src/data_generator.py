import tts, lorem_generator, random, os, csv, math
from pydub import AudioSegment
from datetime import datetime

def generate(phrases=10, phraseLen=None, debug=False):
    now = datetime.now()
    output_file_name = "../resources/data-%d-%d-%d" % (now.hour, now.minute, now.second)
    csv_data = open((output_file_name + ".csv"), 'w', encoding='UTF8')
    writer = csv.writer(csv_data)
    headers = ["speaker", "seconds", "text"]
    writer.writerow(headers)
    if debug == True:
        tts.text_to_speech("Generation started", rate=175)
    mp3_data = AudioSegment.empty()
    for i in range(phrases):
        temp_file_name = "../tmp/tmp-%d.mp3" % random.randint(1000000, 9999999)
        lorem = lorem_generator.generate(phraseLen)
        tts.text_to_mp3(lorem, temp_file_name, voice="male" if i % 2 == 0 else "female", rate=140)
        temp_file = AudioSegment.from_file(temp_file_name)
        mp3_data += temp_file[0:(math.floor(len(temp_file) / 1000)) * 1000]
        writer.writerow(["Male" if i % 2 == 0 else "Female", len(mp3_data) / 1000, lorem])
        os.remove(temp_file_name)
        if debug == True:
            print("Finished %.2f%%\r" % (100 / phrases * (i + 1)), end='')
        
    mp3_data = mp3_data.set_frame_rate(48000)
    mp3_data.export(output_file_name + ".mp3", bitrate="16k", format="mp3")
    if debug == True:
        print("Mp3 data was exported to: %s" % (output_file_name + ".mp3"))
        print("Csv data was exported to: %s" % (output_file_name + ".csv"))
        tts.text_to_speech("Generation completed", rate=175)
    
    return output_file_name + ".mp3", output_file_name + ".csv"
   