class EcgInputError(Exception):

    def __init__(self, checks):
        msg = '; '.join((c['detail'] for c in checks if c['status'] == 'fail')) or 'input tidak valid'
        super().__init__(msg)
        self.checks = checks

def make_check(name, status, detail):
    return {'name': name, 'status': status, 'detail': detail}

def failed(checks):
    return [c for c in checks if c['status'] == 'fail']

def warned(checks):
    return [c for c in checks if c['status'] == 'warn']
