import { Injectable } from '@angular/core';


@Injectable({
  providedIn: 'root'
})
export class ContentService {

  getHeader() {
    return {
      title: 'Dmitry Grinko',
      links: [
        { 
          label: 'About', 
          link: '/about', 
          exact: false,
          submenu: [
            { label: 'Work Experience', link: '/work-experience' },
            { label: 'Technical Skills', link: '/technical-skills' }
          ]
        },
        { label: 'Services', link: '/services', exact: false },
        { label: 'Portfolio', link: '/portfolio', exact: false },
        { label: 'Blog', link: '/blog', exact: false },
        { label: 'Contact', link: '/contact', exact: false },
        { 
          label: 'Hire me', 
          link: '/hire-me', 
          exact: false, 
          isCTA: true,
          submenu: [
            { label: 'Download Resume', link: '/resume' },
            { label: 'Schedule a call', link: '/schedule' }
          ]
        },
      ]
    }
  }

  getFooter() {
    return {
      copyright: {
        startYear: 2015,
        text: 'dmitrygrinko.com'
      },
      socialLinks: [
        {
          icon: 'fab fa-github',
          url: 'https://github.com/dmitrygrinko',
          label: 'GitHub'
        },
        {
          icon: 'fab fa-linkedin',
          url: 'https://www.linkedin.com/in/dmitrygrinko/',
          label: 'LinkedIn'
        }
      ]
    }
  }

}

