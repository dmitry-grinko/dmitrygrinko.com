import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExperienceComponentComponent } from './experience-component.component';

describe('ExperienceComponentComponent', () => {
  let component: ExperienceComponentComponent;
  let fixture: ComponentFixture<ExperienceComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ExperienceComponentComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ExperienceComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
