import json

def new_event(tags, frame, duration):
    return {
        'count': 1,
        'tags': tags,
        'start_frame': frame,
        'start_duration': duration,
        'start_clip': f"{max((duration/1000) - 10, 0):.2f}"
    }

def update_event(event, tags):
    for tag in tags:
        event['tags'][tag] = event['tags'].get(tag, 0) + 1
    event['count'] += 1

class TimelineBuilder:
    """
        Build a timeline of game events based on model output
    """
    
    def __init__(self, killthreshold=4):
        """
            killthreshold minimum number of kills required to register a kill on timeline
        """
        self.logs = []
        self.timeline = []
        self.killthreshold = killthreshold
        # event before it has crossed the threshold, after threshold is passed add it to timeline
        self.tempEvent = None

    def append(self,log, frame, duration):
        self.logs.append(log)
        tags = {}
        if "kill" in log: tags["kill"] = 1
        if "headshot" in log: tags["headshot"] = 1
        if "ace" in log: tags["ace"] = 1
        if len(tags)>0:
            if self.tempEvent:
                update_event(self.tempEvent, tags)
            else:
                self.tempEvent = new_event(tags, frame, duration)
        elif self.tempEvent:
            if self.tempEvent['count'] > self.killthreshold:
                self.tempEvent['end_frame'] = frame
                self.tempEvent['end_duration'] = duration
                self.tempEvent['end_clip'] = f"{duration/1000 + 3:.2f}"
                self.timeline.append(self.tempEvent)
                print(self.tempEvent)
                # self.dump()
            self.tempEvent = None

    def dump(self):
        with open("output.json", "w") as f:
            f.write(
                json.dumps(self.timeline)
            )

    
