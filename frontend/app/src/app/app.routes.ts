import { Routes } from '@angular/router';
import { HomePage } from './pages/home-page/home-page.component';
import { AboutPage } from './pages/about-page/about-page.component';
import { ServicesPage } from './pages/services-page/services-page.component';
import { PortfolioPage } from './pages/portfolio-page/portfolio-page.component';
import { ProjectPage } from './pages/project-page/project-page.component';
import { BlogPage } from './pages/blog-page/blog-page.component';
import { PostPage } from './pages/post-page/post-page.component';
import { LoginPage } from './pages/login-page/login-page.component';
import { RegisterPage } from './pages/register-page/register-page.component';
import { ForgotPasswordPage } from './pages/forgot-password-page/forgot-password-page.component';
import { ResetPasswordPage } from './pages/reset-password-page/reset-password-page.component';
import { VerifyEmailPage } from './pages/verify-email-page/verify-email-page.component';
import { ContactPage } from './pages/contact-page/contact-page.component';
import { ResumePage } from './pages/resume-page/resume-page.component';
import { ScheduleCallPage } from './pages/schedule-call-page/schedule-call-page.component';
import { AuthGuard } from './guards/auth.guard';
import { DefaultLayoutComponent } from './layouts/default-layout/default-layout.component';

export const routes: Routes = [
  {
    path: '',
    component: DefaultLayoutComponent,
    children: [
      { path: '', component: HomePage },
      { path: 'about', component: AboutPage },
      { path: 'services', component: ServicesPage },
      { path: 'portfolio', component: PortfolioPage },
      { path: 'blog', component: BlogPage },
      { path: 'post', component: PostPage },
      { path: 'contact', component: ContactPage },
      { path: 'resume', component: ResumePage },
      { path: 'schedule-call', component: ScheduleCallPage },
    ]
  },
  { 
    path: 'auth',
    children: [
      { path: 'login', component: LoginPage },
      { path: 'register', component: RegisterPage },
      { path: 'forgot-password', component: ForgotPasswordPage },
      { path: 'reset-password', component: ResetPasswordPage },
      { path: 'verify-email', component: VerifyEmailPage },
    ]
  },
  { 
    path: 'project', 
    // component: AuthLayoutComponent,
    children: [
      { path: '', component: ProjectPage },
    ],
    canActivate: [AuthGuard]
  }
];