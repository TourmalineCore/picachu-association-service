import json

from model.model_result_schema import ModelResultSchema


class MessagePacker:
    def __init__(self, model_type):
        self.model_type = model_type

    @staticmethod
    def unpack_the_message_body(message_body):
        message_str = message_body.decode('utf-8')
        message = json.loads(message_str)

        photo_id = message["photo_id"]
        tags = message["tags"]
        tags_without_semicolon_in_objects = []

        for tag in tags:
            tags_without_semicolon_in_objects.append(tag.split(';')[0])

        return photo_id, ' '.join(tags_without_semicolon_in_objects)

    def pack_the_message_body(self, photo_id: int, result):
        valid_result = ModelResultSchema(
            photo_id=photo_id,
            model_type=self.model_type,
            result=result,
        )
        message_body = json.dumps(valid_result.dict()).encode('utf-8')

        return message_body
