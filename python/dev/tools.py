import torch


# def save_model(model, filename):
#     do_save = input('Do you want to save the model (type yes to confirm)? ').lower()
#     if do_save == 'yes':
#         torch.save(model.state_dict(), filename)
#         print('Model saved to %s.' % filename)
#     else:
#         print('Model not saved.')

def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()
