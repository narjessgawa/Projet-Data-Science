from django import forms
from django.contrib.auth.forms import UserCreationForm
from myapp.models import User,Comment,Profile


class CustomUserCreationForm(UserCreationForm):
    face_encoding = forms.FileField(required=False)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('role',)

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = self.cleaned_data['role']

        face_encoding_file = self.cleaned_data['face_encoding']
        if face_encoding_file:
            face_encoding = face_encoding_file.read()
            user.set_face_encoding(face_encoding)

        if commit:
            user.save()
        return user
class CommentForm(forms.ModelForm):
    file = forms.FileField(required=False)  # Add a file field to the form

    def __init__(self, *args, **kwargs):
        professors = kwargs.pop('professors', None)
        students= kwargs.pop('students', None)
        super(CommentForm, self).__init__(*args, **kwargs)
        
        if professors:
            self.fields['destination_user'].queryset = professors
        if students:
            self.fields['destination_user'].queryset = students
    class Meta:
        model = Comment
        fields = ['content', 'source_user','destination_user', 'file']

    def save(self, commit=True):
        comment = super().save(commit=False)
        comment.source_user = self.cleaned_data['source_user']
        comment.destination_user = self.cleaned_data['destination_user']
        comment.content = self.cleaned_data['content']
        if commit:
            comment.save()
        return comment

def save_user_profile(self, instance):
    instance.profile.user_id = self.cleaned_data['user_id']
    # Set other profile fields as needed
    instance.profile.save()
class DatasetForm(forms.Form):
    question = forms.CharField(max_length=255)
    answer = forms.CharField(max_length=255)
    level = forms.CharField(max_length=255)